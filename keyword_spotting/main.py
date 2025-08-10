import argparse
import subprocess

def run_script(name):
    subprocess.run(["python", f"scripts/{name}.py"], check=True)

def main():
    parser = argparse.ArgumentParser(description="Keyword Spotting CLI")
    parser.add_argument("action", choices=["download", "train", "convert", "evaluate"],
                        help="Action to perform: download data, train model, convert to TFLite, or evaluate model")

    args = parser.parse_args()

    if args.action == "download":
        run_script("download_data")
    elif args.action == "train":
        run_script("train")
    elif args.action == "convert":
        run_script("convert_tflite")
    elif args.action == "evaluate":
        run_script("evaluate")

if __name__ == "__main__":
    main()
