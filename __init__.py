from .seedvr2_tiling import SeedVR2_TileSplitter, SeedVR2_TileStitcher, AdvancedColorMatch, CAS_LumaSharpening

NODE_CLASS_MAPPINGS = {
    "SeedVR2_TileSplitter": SeedVR2_TileSplitter,
    "SeedVR2_TileStitcher": SeedVR2_TileStitcher,
    "AdvancedColorMatch": AdvancedColorMatch,
    "CAS_LumaSharpening": CAS_LumaSharpening
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2_TileSplitter": "SeedVR2 Tile Splitter (VRAM Aware)",
    "SeedVR2_TileStitcher": "SeedVR2 Tile Stitcher (Seamless)",
    "AdvancedColorMatch": "Advanced Color Match (OKLAB MKL)",
    "CAS_LumaSharpening": "CAS Luma Sharpening"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
