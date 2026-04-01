import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from gesture_classifier import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SignSpeak gesture classifier")
    parser.add_argument("--data-dir", default="data/samples",              help="Directory with gesture sub-folders")
    parser.add_argument("--output",   default="models/gesture_classifier.pkl", help="Output model path")
    parser.add_argument("--test-size",type=float, default=0.2,             help="Fraction held out for testing")
    args = parser.parse_args()

    results = train(
        data_dir    = args.data_dir,
        output_path = args.output,
        test_size   = args.test_size,
    )

    print(f"\n  Final accuracy: {results['accuracy']:.2%}")
    print(f"  Classes: {results['labels']}")
    print(f"  Total samples used: {results['n_samples']}")
    print("\nRun the app with:\n    python src/app.py\n")
