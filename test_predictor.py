import subprocess

def test_simple():
    cmd = "./run_endtoend.py predictor.yml"
    output = subprocess.run(cmd.split())
    print(output)
    assert output.returncode == 0




