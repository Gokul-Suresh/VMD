import struct

class VmdWriter:
    def __init__(self):
        self.bone_frames = []

    def add_bone_frame(self, name, frame_index, position, rotation):
        """
        name: Japanese bone name (str)
        frame_index: Frame number (int)
        position: (x, y, z) tuple
        rotation: (x, y, z, w) quaternion tuple
        """
        self.bone_frames.append({
            "name": name,
            "frame": frame_index,
            "pos": position,
            "rot": rotation
        })

    def write(self, filename):
        print(f"Writing {len(self.bone_frames)} bone frames to {filename}...")
        with open(filename, "wb") as f:
            # --- 1. Header ---
            # Signature "Vocaloid Motion Data 0002" (30 bytes)
            signature = b"Vocaloid Motion Data 0002"
            f.write(signature.ljust(30, b"\0"))
            
            # Model Name (20 bytes) - We use a generic name
            model_name = "GenerativeAI".encode("shift-jis")
            f.write(model_name.ljust(20, b"\0"))

            # --- 2. Bone Frames ---
            # Count (4 bytes unsigned int)
            f.write(struct.pack("<I", len(self.bone_frames)))

            # Data
            for frame in self.bone_frames:
                # Bone Name (15 bytes, Shift-JIS encoded)
                try:
                    name_bytes = frame["name"].encode("shift-jis")
                except UnicodeEncodeError:
                    # Fallback if name is weird, though it shouldn't be
                    name_bytes = frame["name"].encode("utf-8")
                f.write(name_bytes.ljust(15, b"\0"))
                
                # Frame Number (4 bytes int)
                f.write(struct.pack("<I", frame["frame"]))
                
                # Position (3 floats: x, y, z)
                px, py, pz = frame["pos"]
                f.write(struct.pack("<3f", px, py, pz))
                
                # Rotation (4 floats: x, y, z, w) - Quaternion
                rx, ry, rz, rw = frame["rot"]
                f.write(struct.pack("<4f", rx, ry, rz, rw))
                
                # Interpolation Curve (64 bytes)
                # We use default linear interpolation (filled with 0s and distinct curve points)
                # MMD is picky, but all-zeros usually defaults to linear or stiff.
                # A standard default curve:
                interpolation = b'\x00' * 64
                f.write(interpolation)

            # ---