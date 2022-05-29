import os
import zipfile


def zip_compress(filename: str) -> list:
    f = zipfile.ZipFile(filename[:-3] + "zip", 'w')
    f.write(filename,
            compress_type=zipfile.ZIP_DEFLATED)
    before_size = os.stat(filename).st_size
    after_zip_size = os.stat(filename[:-3] + "zip").st_size
    print("before_size: ", round(before_size / 1024, 2), "kb")
    print("after_zip_size: ", round(after_zip_size / 1024, 2), " kb")
    return [round(before_size / 1024, 2), round(after_zip_size / 1024, 2)]
