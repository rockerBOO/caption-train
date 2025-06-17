FROM huggingface/transformers-pytorch-gpu:latest

WORKDIR /app

COPY requirements.txt .

# pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels torch torchvision  --index-url https://download.pytorch.org/whl/cu118 && \
RUN python3 -m pip install --no-cache-dir --upgrade pip && \ 
		pip install --no-cache-dir --no-deps -r requirements.txt
#
# FROM huggingface/transformers-pytorch-gpu:latest
#
#
# COPY --from=builder /app/wheels /wheels
# COPY --from=builder /app/requirements.txt .
#
# RUN pip install --no-cache /app/wheels/*


RUN addgroup --system app && adduser --system --group app
RUN chown app /app
USER app

COPY train2.py train2.py

# RUN mkdir /app/.cache

ENTRYPOINT ["python3"]

CMD ["/app/train2.py"]
