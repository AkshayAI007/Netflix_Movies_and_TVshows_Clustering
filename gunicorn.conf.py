import os

bind        = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
workers     = 1
timeout     = 300   # model loads in background thread; this covers slow Render CPUs
keepalive   = 5
