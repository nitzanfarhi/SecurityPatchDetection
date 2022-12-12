
def safe_mkdir(dirname):
    with contextlib.suppress(FileExistsError):
        os.mkdir(dirname)