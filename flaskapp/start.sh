
#!/bin/bash

# Start the Flask application with Gunicorn
gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 2
