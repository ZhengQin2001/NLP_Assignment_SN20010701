import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|evaluate]")
        return

    command = sys.argv[1]

    if command == "train":
        import train
    elif command == "evaluate":
        import evaluate
    else:
        print("Invalid command. Use 'train' or 'evaluate'.")

if __name__ == "__main__":
    main()
