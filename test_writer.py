from vmd_writer import VmdWriter

writer = VmdWriter()
# Add a fake frame: Frame 0, Center Bone, No movement, No rotation
writer.add_bone_frame("センター", 0, (0,0,0), (0,0,0,1))

writer.write("test.vmd")