import torch
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Show the keys of a PyTorch saved file."
    )
    parser.add_argument("filename", type=str, help="Path to the saved PyTorch file")
    args = parser.parse_args()

    try:
        # Load the saved model file
        loaded_dict = torch.load(args.filename, map_location=torch.device("cpu"))['state_dict']

        if isinstance(loaded_dict, dict):
            # Display the keys of the dictionary
            print("Keys in the saved model file:")
            for key in loaded_dict.keys():
                print(key)
        else:
            print("The loaded file is not a dictionary.")
    except Exception as e:
        print(f"An error occurred while loading the model file: {e}")


if __name__ == "__main__":
    main()
