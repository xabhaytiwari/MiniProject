from sentence_transformers import SentenceTransformer, util
from PIL import Image

model = SentenceTransformer("sentence-transformers/clip-ViT-B-32")

cat_img_emb_1 = model.encode(Image.open('PetImages/Cat/1.jpg'))
cat_img_emb_2 = model.encode(Image.open('PetImages/Cat/4.jpg'))
dog_img_emb = model.encode(Image.open('PetImages/Dog/1.jpg'))

cos_score = util.cos_sim(cat_img_emb_1, cat_img_emb_2)
print(cos_score)
