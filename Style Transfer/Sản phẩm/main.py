from fastapi import FastAPI, File, UploadFile,HTTPException, status 
from fastapi.responses import FileResponse
from style_transfer_model import StyleTransfer


app = FastAPI()

@app.post("/style-transfer/")
async def style_transfer(file1: UploadFile = File(..., description="Style image (JPG or PNG)"), file2: UploadFile = File(..., description="Content image (JPG or PNG)")):
    # if not file1 or not file2:
    #     raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Bạn cần cung cấp đủ 2 tệp Style image và Content image")
    if not is_image_file(file1.filename):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tệp Style image phải là hình ảnh JPG hoặc PNG.")
    if not is_image_file(file2.filename):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tệp Content image phải là hình ảnh JPG hoặc PNG.")
    st = StyleTransfer(file1.file, file2.file,epochs=5,steps_per_epoch=50)
    st.run_gradient()
    st.save_image(img_format = 'jpg')

@app.get("/style-transfer/")
async def get_image(): 
    return FileResponse('stylized-image.jpg', media_type='image/jpeg')


def is_image_file(filename: str) -> bool:
    return filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png")
#uvicorn main:app --reload