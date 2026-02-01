
import torch
import torchvision.models as models
import time


# Detect the best available device
if torch.cuda.is_available():
    device = 'cuda'
    device_type = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
    device_type = 'cpu'  # MPS doesn't support autocast yet, use cpu type
else:
    device = 'cpu'
    device_type = 'cpu'

print(f"Using device: {device}")

def benchmark(model, input_tensor, steps=100, label="NCHW"):
    # 1. Warmup (to initialize kernels and caches)
    print(f"[{label}] Warming up...")
    
    # Use autocast only if supported
    if device_type in ['cuda', 'cpu']:
        with torch.amp.autocast(device_type):
            for _ in range(10):
                _ = model(input_tensor)
    else:
        for _ in range(10):
            _ = model(input_tensor)
    
    # Synchronize based on device
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()
    
    # 2. Measurement
    print(f"[{label}] Benchmarking...")
    start_time = time.time()
    
    if device_type in ['cuda', 'cpu']:
        with torch.amp.autocast(device_type):
            for _ in range(steps):
                _ = model(input_tensor)
    else:
        for _ in range(steps):
            _ = model(input_tensor)
    
    # Synchronize based on device
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / steps * 1000 # convert to ms
    print(f"[{label}] Average time per batch: {avg_time:.2f} ms")
    return avg_time

def run_test():
    # Settings
    BATCH_SIZE = 64
    STEPS = 50
    
    # Setup Model (ResNet50 is a classic CNN) and Data
    print(f"Setting up ResNet50 with Batch Size {BATCH_SIZE}...")
    model = models.resnet50().to(device).eval()
    data = torch.randn(BATCH_SIZE, 3, 224, 224).to(device)

    # --- TEST 1: Default (NCHW) ---
    # PyTorch defaults to NCHW. No changes needed.
    time_nchw = benchmark(model, data, steps=STEPS, label="Default (NCHW)")

    # --- TEST 2: Channels Last (NHWC) ---
    # Convert both Model AND Data to Channels Last memory format
    model_cl = model.to(memory_format=torch.channels_last)
    data_cl = data.to(memory_format=torch.channels_last)
    
    time_nhwc = benchmark(model_cl, data_cl, steps=STEPS, label="Channels Last")

    # --- Results ---
    speedup = (time_nchw - time_nhwc) / time_nchw * 100
    print("-" * 40)
    print(f"Speedup: {speedup:.2f}%")
    print("-" * 40)

if __name__ == "__main__":
    run_test()