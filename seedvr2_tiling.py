import torch
import torch.nn.functional as F

class SeedVR2_TileSplitter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1}),
                "overlap": ("INT", {"default": 128, "min": 0, "max": 512, "step": 8}),
                "noise_injection": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01}),
                "vram_mode": (["Auto", "Low (8GB)", "Medium (12GB)", "High (16GB+)"], {"default": "Auto"}),
            }
        }
        
    RETURN_TYPES = ("IMAGE", "INT", "INT", "TILE_INFO",)
    RETURN_NAMES = ("tiles", "tile_resolution", "tile_max_resolution", "tile_info",)
    FUNCTION = "split_image"
    CATEGORY = "SeedVR2_Tiling"

    def split_image(self, image, upscale_by, overlap, noise_injection, vram_mode):
        B, H, W, C = image.shape
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        
        # Determine tile size based on VRAM
        tile_size_mp = 1.2 # Default medium
        if vram_mode == "Auto":
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if total_memory < 8.5:
                    tile_size_mp = 0.75
                elif total_memory <= 12.5:
                    tile_size_mp = 1.2
                else:
                    tile_size_mp = 2.0
        elif vram_mode == "Low (8GB)":
            tile_size_mp = 0.75
        elif vram_mode == "Medium (12GB)":
            tile_size_mp = 1.2
        elif vram_mode == "High (16GB+)":
            tile_size_mp = 2.0
            
        # Calculate base tile side, assure multiple of 8
        base_side = int((tile_size_mp * 1000000) ** 0.5)
        base_side = (base_side // 8) * 8
        
        stride = base_side - overlap
        if stride <= 0:
            stride = 64
            
        import math
        rows = math.ceil((H - overlap) / stride) if H > base_side else 1
        cols = math.ceil((W - overlap) / stride) if W > base_side else 1
        
        tile_h = base_side if rows > 1 else math.ceil(H / 8) * 8
        tile_w = base_side if cols > 1 else math.ceil(W / 8) * 8
        
        pad_h = (rows * stride + overlap) - H if rows > 1 else tile_h - H
        pad_w = (cols * stride + overlap) - W if cols > 1 else tile_w - W
        
        pad_h = max(0, int(pad_h))
        pad_w = max(0, int(pad_w))
        
        img_permuted = image.permute(0, 3, 1, 2) # (B, C, H, W)
        try:
            img_padded = F.pad(img_permuted, (0, pad_w, 0, pad_h), mode='reflect')
        except RuntimeError:
            img_padded = F.pad(img_permuted, (0, pad_w, 0, pad_h), mode='replicate')
        img_padded = img_padded.permute(0, 2, 3, 1) # back to (B, H', W', C)
        
        H_pad, W_pad = img_padded.shape[1], img_padded.shape[2]
        
        tiles = []
        positions = []
        
        for b in range(B):
            for r in range(rows):
                for c in range(cols):
                    y0 = r * stride
                    y1 = y0 + base_side
                    x0 = c * stride
                    x1 = x0 + base_side
                    
                    y1 = min(y1, H_pad)
                    x1 = min(x1, W_pad)
                    
                    tile = img_padded[b:b+1, y0:y1, x0:x1, :]
                    
                    if noise_injection > 0:
                        # Calculate luminance to protect pure blacks from noise
                        if C >= 3:
                            luminance = 0.299 * tile[..., 0:1] + 0.587 * tile[..., 1:2] + 0.114 * tile[..., 2:3]
                        else:
                            luminance = tile[..., 0:1] # Fallback for grayscale
                        
                        # Scale noise by luminance (dark areas get less/no noise)
                        noise = torch.randn_like(tile) * noise_injection * luminance
                        tile = tile + noise
                        
                    tiles.append(tile)
                    positions.append((b, r, c, y0, y1, x0, x1))
                    
        batch_tiles = torch.cat(tiles, dim=0).to(dtype)
        
        tile_info = {
            "original_shape": (B, H, W, C),
            "padded_shape": (B, H_pad, W_pad, C),
            "base_side": base_side,
            "overlap": overlap,
            "stride": stride,
            "rows": rows,
            "cols": cols,
            "positions": positions,
            "upscale_by": upscale_by,
            "dtype": str(dtype),
            "tile_h": tile_h,
            "tile_w": tile_w
        }
        
        # Calculate optimal output resolution for Numz node to prevent automatic resizing
        tile_resolution = int(min(tile_h, tile_w) * upscale_by)
        tile_max_resolution = int(max(tile_h, tile_w) * upscale_by)
        
        return (batch_tiles, tile_resolution, tile_max_resolution, tile_info)

class SeedVR2_TileStitcher:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "tile_info": ("TILE_INFO",),
                "black_level_fix": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.2, "step": 0.01}),
            }
        }
        
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "stitch_tiles"
    CATEGORY = "SeedVR2_Tiling"

    def gaussian_pyramid(self, img, levels):
        pyr = [img]
        for _ in range(levels - 1):
            down = F.avg_pool2d(pyr[-1].permute(0, 3, 1, 2), kernel_size=2, stride=2).permute(0, 2, 3, 1)
            pyr.append(down)
        return pyr

    def laplacian_pyramid(self, img, levels):
        g_pyr = self.gaussian_pyramid(img, levels)
        l_pyr = []
        for i in range(levels - 1):
            up = F.interpolate(g_pyr[i+1].permute(0, 3, 1, 2), size=(g_pyr[i].shape[1], g_pyr[i].shape[2]), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            l_pyr.append(g_pyr[i] - up)
        l_pyr.append(g_pyr[-1])
        return l_pyr

    def collapse_pyramid(self, l_pyr):
        img = l_pyr[-1]
        for i in range(len(l_pyr)-2, -1, -1):
            up = F.interpolate(img.permute(0, 3, 1, 2), size=(l_pyr[i].shape[1], l_pyr[i].shape[2]), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            img = l_pyr[i] + up
        return img

    def stitch_tiles(self, tiles, tile_info, black_level_fix=0.05):
        tiles = tiles.cpu().float()
        
        B_orig, H_orig, W_orig, C = tile_info["original_shape"]
        B_pad, H_pad, W_pad, C_pad = tile_info["padded_shape"]
        upscale_by = tile_info["upscale_by"]
        
        H_out = int(H_orig * upscale_by)
        W_out = int(W_orig * upscale_by)
        H_pad_out = int(H_pad * upscale_by)
        W_pad_out = int(W_pad * upscale_by)
        
        overlap_out = int(tile_info["overlap"] * upscale_by)
        tile_h_out = int(tile_info.get("tile_h", tile_info["base_side"]) * upscale_by)
        tile_w_out = int(tile_info.get("tile_w", tile_info["base_side"]) * upscale_by)
        
        rows = tile_info["rows"]
        cols = tile_info["cols"]
        positions = tile_info["positions"]
        
        levels = 4
        
        out_pyr = []
        weight_pyr = []
        
        for i in range(levels):
            scale = 2 ** i
            h_lvl = H_pad_out // scale
            w_lvl = W_pad_out // scale
            out_pyr.append(torch.zeros((B_orig, h_lvl, w_lvl, C), dtype=torch.float32, device='cpu'))
            weight_pyr.append(torch.zeros((B_orig, h_lvl, w_lvl, 1), dtype=torch.float32, device='cpu'))

        y_idx = torch.arange(tile_h_out, dtype=torch.float32)
        x_idx = torch.arange(tile_w_out, dtype=torch.float32)
        dist_y = torch.minimum(y_idx, tile_h_out - 1 - y_idx)
        dist_x = torch.minimum(x_idx, tile_w_out - 1 - x_idx)
        
        mask_y = torch.clamp(dist_y / (overlap_out / 2 + 1e-5), 0, 1)
        mask_x = torch.clamp(dist_x / (overlap_out / 2 + 1e-5), 0, 1)
        mask = torch.outer(mask_y, mask_x).unsqueeze(0).unsqueeze(-1)
        mask_pyr = self.gaussian_pyramid(mask, levels)

        for i, (b, r, c, y0, y1, x0, x1) in enumerate(positions):
            tile = tiles[i:i+1]
            
            y0_out = int(y0 * upscale_by)
            y1_out = int(y1 * upscale_by)
            x0_out = int(x0 * upscale_by)
            x1_out = int(x1 * upscale_by)
            
            t_l_pyr = self.laplacian_pyramid(tile, levels)
            
            for lvl in range(levels):
                scale = 2 ** lvl
                y0_lvl = y0_out // scale
                
                # Use actual shape for h/w slices instead of trying to map coordinates
                t_lvl_h = t_l_pyr[lvl].shape[1]
                t_lvl_w = t_l_pyr[lvl].shape[2]
                
                y1_lvl = y0_lvl + t_lvl_h
                x0_lvl = x0_out // scale
                x1_lvl = x0_lvl + t_lvl_w
                
                m_lvl = mask_pyr[lvl]
                h_slice = min(t_lvl_h, m_lvl.shape[1])
                w_slice = min(t_lvl_w, m_lvl.shape[2])
                
                y1_lvl = y0_lvl + h_slice
                x1_lvl = x0_lvl + w_slice
                
                t_patch = t_l_pyr[lvl][:, :h_slice, :w_slice, :]
                m_patch = m_lvl[:, :h_slice, :w_slice, :]
                
                out_pyr[lvl][b:b+1, y0_lvl:y1_lvl, x0_lvl:x1_lvl, :] += t_patch * m_patch
                weight_pyr[lvl][b:b+1, y0_lvl:y1_lvl, x0_lvl:x1_lvl, :] += m_patch

        for lvl in range(levels):
            weight = weight_pyr[lvl]
            weight[weight == 0] = 1.0
            out_pyr[lvl] /= weight

        final_img_padded = self.collapse_pyramid(out_pyr)
        final_img = final_img_padded[:, :H_out, :W_out, :]
        
        # Apply black level fix
        if black_level_fix > 0.0:
            # Shift the black level down and rescale
            final_img = torch.clamp(final_img - black_level_fix, min=0.0)
            final_img = final_img / (1.0 - black_level_fix)
            
        final_img = torch.clamp(final_img, 0.0, 1.0)
        
        return (final_img,)

class AdvancedColorMatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "target": ("IMAGE",),
                "source": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "preserve_luma": ("BOOLEAN", {"default": False}),
            }
        }
        
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "match_color"
    CATEGORY = "SeedVR2_Tiling"

    @staticmethod
    def srgb_to_linear(rgb):
        return torch.where(rgb <= 0.04045, rgb / 12.92, torch.pow((rgb + 0.055) / 1.055, 2.4))

    @staticmethod
    def linear_to_srgb(linear):
        return torch.where(linear <= 0.0031308, 12.92 * linear, 1.055 * torch.pow(linear.clamp(min=1e-8), 1.0 / 2.4) - 0.055)

    @staticmethod
    def srgb_to_oklab(rgb):
        # 1. Apply sRGB transfer function
        linear = AdvancedColorMatch.srgb_to_linear(rgb)
        
        # Matrix to convert Linear sRGB to cone responses (LMS)
        m1 = torch.tensor([
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005]
        ], dtype=linear.dtype, device=linear.device)
        
        lms = torch.matmul(linear, m1.T)
        
        # 2. Non-linearity (cube root)
        lms_ = torch.sign(lms) * torch.pow(torch.abs(lms), 1.0/3.0)
        
        # Matrix to convert non-linear LMS to OKLAB
        m2 = torch.tensor([
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660]
        ], dtype=linear.dtype, device=linear.device)
        
        lab = torch.matmul(lms_, m2.T)
        return lab

    @staticmethod
    def oklab_to_srgb(lab):
        # Matrix to convert OKLAB to non-linear LMS
        m1_inv = torch.tensor([
            [1.0000000000, 0.3963377774, 0.2158037573],
            [1.0000000000, -0.1055613458, -0.0638541728],
            [1.0000000000, -0.0894841775, -1.2914855480]
        ], dtype=lab.dtype, device=lab.device)
        
        lms_ = torch.matmul(lab, m1_inv.T)
        
        # 2. Inverse non-linearity (cube)
        lms = torch.pow(lms_, 3.0)
        
        # Matrix to convert LMS to Linear sRGB
        m2_inv = torch.tensor([
            [4.0767416621, -3.3077115913, 0.2309699292],
            [-1.2684380046, 2.6097574011, -0.3413193965],
            [-0.0041960863, -0.7034186147, 1.7076147010]
        ], dtype=lab.dtype, device=lab.device)
        
        linear = torch.matmul(lms, m2_inv.T)
        linear = torch.clamp(linear, 0.0, 1.0)
        
        # 3. Apply inverse sRGB transfer function
        rgb = AdvancedColorMatch.linear_to_srgb(linear)
        return torch.clamp(rgb, 0.0, 1.0)

    @staticmethod
    def match_color_mkl(source_features, target_features, preserve_luma):
        # Monge-Kantorovich Linear (MKL) Optimal Transport mapping
        # Expects features as [N, C] (flattened images)

        if preserve_luma:
            # Only match 'a' and 'b' channels (indices 1 and 2)
            channels_to_match = [1, 2]
            matched_features = target_features.clone()
        else:
            # Match all 'L', 'a', 'b' channels
            channels_to_match = [0, 1, 2]
            matched_features = target_features.clone()

        src = source_features[:, channels_to_match]
        tgt = target_features[:, channels_to_match]

        # 1. Means
        mu_src = torch.mean(src, dim=0, keepdim=True)
        mu_tgt = torch.mean(tgt, dim=0, keepdim=True)

        # Center data
        src_centered = src - mu_src
        tgt_centered = tgt - mu_tgt

        # 2. Covariance matrices
        # N can be very large, calculate covariance efficiently
        N_src = src.shape[0]
        N_tgt = tgt.shape[0]
        
        cov_src = torch.mm(src_centered.t(), src_centered) / max(1, N_src - 1)
        cov_tgt = torch.mm(tgt_centered.t(), tgt_centered) / max(1, N_tgt - 1)

        # 3. MKL Transformation: T(x) = mu_src + C * (x - mu_tgt)
        # where C = cov_src^(1/2) * cov_tgt^(-1/2) (approximated for numerical stability via SVD)
        
        try:
            u_s, s_s, v_s = torch.svd(cov_src)
            cov_src_root = torch.mm(u_s, torch.mm(torch.diag(torch.sqrt(torch.abs(s_s) + 1e-8)), v_s.t()))
            
            u_t, s_t, v_t = torch.svd(cov_tgt)
            cov_tgt_inv_root = torch.mm(u_t, torch.mm(torch.diag(1.0 / torch.sqrt(torch.abs(s_t) + 1e-8)), v_t.t()))
            
            C_matrix = torch.mm(cov_src_root, cov_tgt_inv_root)
        except RuntimeError: # SVD fallback if it fails to converge
            C_matrix = torch.eye(len(channels_to_match), dtype=src.dtype, device=src.device)

        # 4. Apply mapping
        matched_sub = torch.mm(tgt_centered, C_matrix.t()) + mu_src
        
        if preserve_luma:
            matched_features[:, 1:] = matched_sub
        else:
            matched_features = matched_sub

        return matched_features

    def match_color(self, target, source, strength=1.0, preserve_luma=False):
        if strength == 0.0:
            return (target,)
            
        # Target usually has shape (B, H, W, 3) in ComfyUI
        B_t, H_t, W_t, C_t = target.shape
        B_s, H_s, W_s, C_s = source.shape
        
        # Verify both are RGB (3 channels) or drop alpha if RGBA
        tgt_rgb = target[..., :3].to(torch.float32)
        src_rgb = source[..., :3].to(torch.float32)
        
        matched_batch = []
        for b in range(B_t):
            # If target has more batches than source, use the last source image
            sb = min(b, B_s - 1)
            
            # Convert to OKLAB
            tgt_oklab = self.srgb_to_oklab(tgt_rgb[b])
            src_oklab = self.srgb_to_oklab(src_rgb[sb])
            
            # Flatten to shape (N, 3)
            tgt_flat = tgt_oklab.reshape(-1, 3)
            src_flat = src_oklab.reshape(-1, 3)
            
            # Apply MKL Match
            matched_flat = self.match_color_mkl(src_flat, tgt_flat, preserve_luma)
            
            # Interpolate based on strength
            if strength < 1.0:
                matched_flat = (matched_flat * strength) + (tgt_flat * (1.0 - strength))
                
            # Reshape back to (H, W, 3)
            matched_oklab = matched_flat.reshape(H_t, W_t, 3)
            
            # Convert back to sRGB
            matched_rgb = self.oklab_to_srgb(matched_oklab)
            matched_batch.append(matched_rgb.unsqueeze(0))
            
        final_target = torch.cat(matched_batch, dim=0)
        
        # If original target had an alpha channel, restore it
        if C_t == 4:
            alpha = target[..., 3:4]
            final_target = torch.cat([final_target, alpha], dim=-1)
            
        return (final_target,)

class CAS_LumaSharpening:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }
        
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sharpen"
    CATEGORY = "SeedVR2_Tiling"

    @staticmethod
    def rgb_to_ycbcr(rgb):
        # BT.709 conversion
        matrix = torch.tensor([
            [ 0.2126,  0.7152,  0.0722],
            [-0.1146, -0.3854,  0.5000],
            [ 0.5000, -0.4542, -0.0458]
        ], dtype=rgb.dtype, device=rgb.device)
        
        ycbcr = torch.matmul(rgb, matrix.T)
        offset = torch.tensor([0.0, 0.5, 0.5], dtype=rgb.dtype, device=rgb.device)
        return ycbcr + offset

    @staticmethod
    def ycbcr_to_rgb(ycbcr):
        offset = torch.tensor([0.0, 0.5, 0.5], dtype=ycbcr.dtype, device=ycbcr.device)
        ycbcr_shifted = ycbcr - offset
        
        matrix = torch.tensor([
            [1.0000,  0.0000,  1.5748],
            [1.0000, -0.1873, -0.4681],
            [1.0000,  1.8556,  0.0000]
        ], dtype=ycbcr.dtype, device=ycbcr.device)
        
        rgb = torch.matmul(ycbcr_shifted, matrix.T)
        return torch.clamp(rgb, 0.0, 1.0)

    @staticmethod
    def apply_cas(luma, amount):
        # Luma expected as [B, 1, H, W]
        B, C, H, W = luma.shape
        
        # Ensure we don't divide by zero
        img = torch.clamp(luma, 1e-5, 1.0)
        
        # Pad replication for border handling without artifacts
        img_pad = F.pad(img, (1, 1, 1, 1), mode='replicate')
        
        # Neighbors: a (top), b (left), c (bottom), d (right), e (center)
        a = img_pad[:, :, :-2, 1:-1]
        b = img_pad[:, :, 1:-1, :-2]
        c = img_pad[:, :, 2:, 1:-1]
        d = img_pad[:, :, 1:-1, 2:]
        e = img[:, :, :, :]
        
        # Optional: diagonals for more precise minimum/maximum estimation (slower but better)
        # f = img_pad[:, :, :-2, :-2] # top-left
        # g = img_pad[:, :, :-2, 2:]  # top-right
        # h = img_pad[:, :, 2:, :-2]  # bottom-left
        # i = img_pad[:, :, 2:, 2:]   # bottom-right

        # CAS Algorithm step 1: min/max
        min_rg = torch.minimum(a, b)
        min_rg = torch.minimum(min_rg, c)
        min_rg = torch.minimum(min_rg, d)
        min_rg = torch.minimum(min_rg, e)
        
        max_rg = torch.maximum(a, b)
        max_rg = torch.maximum(max_rg, c)
        max_rg = torch.maximum(max_rg, d)
        max_rg = torch.maximum(max_rg, e)
        
        # CAS Algorithm step 2: Calculate weight (w)
        # Weight formula: w = -1 / (amount * max(min(min_rg, 1-max_rg)/max_rg, 1e-5)) (simplified AMD formula)
        # Actually AMD formula: 
        # w = 1.0 / (-8.0 + (3.0 * amount)) # basic curve
        # Adaptive amp:
        
        # Scale user amount to match 1.8 max intensity
        scaled_amount = amount * 1.8
        
        amp = torch.clamp(torch.minimum(min_rg, 1.0 - max_rg) / max_rg, 0.0, 1.0)
        w = amp * (scaled_amount * -0.125) # Scale sharpness based on contrast amp
        
        # Apply filter:  (a + b + c + d) * w + e / (4 * w + 1)
        res = (a + b + c + d) * w + e
        res = res / (4.0 * w + 1.0)
        
        return torch.clamp(res, 0.0, 1.0)

    def sharpen(self, image, amount=0.8):
        if amount == 0.0:
            return (image,)
            
        B, H, W, C = image.shape
        rgb = image[..., :3].to(torch.float32)
        
        # 1. Convert to YCbCr
        ycbcr = self.rgb_to_ycbcr(rgb)
        
        # 2. Extract Luma (Y) and permute to NCHW for CAS [B, 1, H, W]
        luma = ycbcr[..., 0:1].permute(0, 3, 1, 2)
        
        # 3. Apply CAS to Luma only
        sharpened_luma = self.apply_cas(luma, amount)
        
        # 4. Put back Luma and convert to RGB
        ycbcr[..., 0:1] = sharpened_luma.permute(0, 2, 3, 1)
        final_rgb = self.ycbcr_to_rgb(ycbcr)
        
        # Restore Alpha if present
        if C == 4:
            alpha = image[..., 3:4]
            final_rgb = torch.cat([final_rgb, alpha], dim=-1)
            
        return (final_rgb,)
