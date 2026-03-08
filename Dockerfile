FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Create workspace inside the container
RUN mkdir /job
WORKDIR /job

# Mount points:
# - /job/src   : source code (your 'src' directory)
# - /job/work  : model checkpoints (must contain 'byt5-finetuned')
# - /job/data  : input data (contains input.txt at grading time)
# - /job/output: where predictions (pred.txt) will be written
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# Install Python dependencies needed for inference
RUN pip install --no-cache-dir transformers sentencepiece

# Default command does nothing; the grader will override this with:
#   bash /job/src/predict.sh /job/data/input.txt /job/output/pred.txt
