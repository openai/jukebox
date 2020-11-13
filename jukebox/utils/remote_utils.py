import sys
import subprocess

def download(remote_path, local_path, async_download=False):
    args = ['wget', '-O', local_path, remote_path]
    print("Running ", " ".join(args))
    if async_download:
        subprocess.Popen(args)
    else:
        subprocess.call(args)

# GCE
def gs_download(gs_path, local_path, async_download=False):
    args = ['gsutil',
            '-o', 'GSUtil:parallel_thread_count=1',
            '-o', 'GSUtil:sliced_object_download_max_components=8',
            'cp', gs_path, local_path]
    if async_download:
        subprocess.Popen(args)
    else:
        subprocess.call(args)


def gs_upload(local_path, gs_path, async_upload=False):
    # NOTE: Download and upload have differ -o flags.
    # We also use -n to prevent clobbering checkpoints by mistake
    assert not local_path.startswith("gs://")
    assert gs_path.startswith("gs://")
    args = ['gsutil',
            '-o', 'GSUtil:parallel_composite_upload_threshold=150M',
            'cp', '-n', local_path, gs_path]
    if async_upload:
        subprocess.Popen(args)
    else:
        subprocess.call(args)

def ls(regex):
    outputs = subprocess.check_output(['gsutil', 'ls', regex]).decode(sys.stdout.encoding)
    outputs = outputs.split('\n')
    outputs = [output for output in outputs if output is not '']
    return outputs

