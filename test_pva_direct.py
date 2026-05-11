import pvaccess as pva
import time
import numpy as np

channel_name = "12idBFS1:Pva1:Image"

print(f"Creating channel {channel_name}")
ch = pva.Channel(channel_name, pva.PVA)

# Define a simple callback
def callback(pv):
    print(f"Callback received: {type(pv)}")
    if hasattr(pv, 'keys'):
        print(f"  Keys: {list(pv.keys())}")
        if 'value' in pv:
            val = pv['value']
            print(f"  Value type: {type(val)}")
            if isinstance(val, (list, tuple)) and len(val) > 0:
                print(f"  First element type: {type(val[0])}")
                if hasattr(val[0], 'dtype'):
                    print(f"  First element dtype: {val[0].dtype}")
                if hasattr(val[0], 'shape'):
                    print(f"  First element shape: {val[0].shape}")
                    
                # Try to convert to numpy array like our extractor does
                try:
                    # This mimics what _extract_ntndarray does
                    flat = np.frombuffer(bytes(val[0]), dtype=val[0].dtype) if isinstance(
                        val[0], (bytes, bytearray, memoryview)
                    ) else np.asarray(val[0], dtype=val[0].dtype)
                    print(f"  Converted to numpy array: shape={flat.shape}, dtype={flat.dtype}")
                    
                    # Try to get dimensions
                    if 'dimension' in pv:
                        dims = [d["size"] for d in pv["dimension"]]
                        print(f"  Dimensions from pv: {dims}")
                        if len(dims) == 2:
                            w, h = dims
                            reshaped = flat.reshape((h, w))
                            print(f"  Reshaped to (h,w): {reshaped.shape}")
                        elif len(dims) == 3:
                            a, b, c = dims
                            print(f"  3D dimensions: {a},{b},{c}")
                            # Try channels-last first
                            try:
                                reshaped = flat.reshape((b, a, c))
                                print(f"  Reshaped to channels-last (b,a,c): {reshaped.shape}")
                            except:
                                # Try channels-first
                                try:
                                    reshaped = flat.reshape((c, b, a))
                                    print(f"  Reshaped to channels-first (c,b,a): {reshaped.shape}")
                                except:
                                    print(f"  Could not reshape 3D array")
                    else:
                        print(f"  No 'dimension' key in pv")
                except Exception as e:
                    print(f"  Error converting to numpy: {e}")
            else:
                print(f"  Value is empty or not a list/tuple")
        else:
            print(f"  No 'value' key in pv")
    else:
        print(f"  pv has no keys attribute")
        # Try to see what it is
        print(f"  pv repr: {repr(pv)[:500]}")

print("Subscribing...")
ch.subscribe("test", callback)
ch.startMonitor()

print("Waiting for frames (will timeout after 30 seconds)...")
start_time = time.time()
while time.time() - start_time < 30:
    time.sleep(0.1)  # Just sleep, callback will happen in pvaccess thread

print("Time's up. Stopping monitor...")
ch.stopMonitor()
ch.unsubscribe("test")
print("Done.")