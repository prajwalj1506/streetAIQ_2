import pinggy
import time

try:
    print("Starting Pinggy tunnel to port 5173...")
    # start_tunnel typically runs the tunnel in the background
    tunnel = pinggy.start_tunnel(forwardto=5173)
    print("Tunnel started successfully!")
    
    # Try to extract the public URL
    # We will search the tunnel object properties
    for attr in dir(tunnel):
        if 'url' in attr.lower():
            val = getattr(tunnel, attr)
            print(f"Property '{attr}': {val}")
            if callable(val):
                try:
                    print(f"Callable '{attr}' returned: {val()}")
                except Exception:
                    pass
except Exception as e:
    print("Error during tunnel start:", e)
