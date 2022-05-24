import zipfile
import os

if __name__ == '__main__':
    f = zipfile.ZipFile('output_compressed_trajectory.zip', 'w')
    f.write('output_compressed_trajectory.txt',
            compress_type=zipfile.ZIP_DEFLATED)
    size = os.stat("output_compressed_trajectory.zip").st_size
    print(size)
    print(round(size / 1024.0, 2))
