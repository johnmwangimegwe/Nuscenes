import multiprocessing as mp
from test_fusion import evaluate_model

def run_demo():
    print("Running three-modal fusion demo...")
    evaluate_model()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    run_demo()