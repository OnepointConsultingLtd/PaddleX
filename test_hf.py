from paddlex import create_pipeline
pipeline = create_pipeline(
    pipeline="paddlex/configs/pipelines/PaddleOCR-VL-HF.yaml"
)
for result in pipeline.predict("GLM-4.5V.pdf"):
    print(result.markdown)