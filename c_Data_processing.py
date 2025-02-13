def Image_nparray(id1):
    byte_string = train_h5py[id1][()]
    nparr = np.frombuffer(byte_string, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_resized = cv2.resize(image, (128, 128))
    image_vector = image_resized.astype(np.float32) / 255.0
    return image_vector


images = []

for i in range(len(df)):
    isic_id = df.isic_id.iloc[i]  # Use the current index
    image_array = Image_nparray(isic_id)
    images.append(image_array)
df["image"] = images

# Categorical features which will be one hot encoded
CATEGORICAL_COLUMNS = ["sex", "anatom_site_general",
            "tbp_tile_type","tbp_lv_location", ]
Primary_key=["isic_id","image","target"]

# Numeraical features which will be normalized
NUMERIC_COLUMNS = ["age_approx", "tbp_lv_nevi_confidence", "clin_size_long_diam_mm",
           "tbp_lv_areaMM2", "tbp_lv_area_perim_ratio", "tbp_lv_color_std_mean",
           "tbp_lv_deltaLBnorm", "tbp_lv_minorAxisMM", ]

FEAT_COLS = Primary_key + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS

df=df.drop(columns=["isic_id"])

from sklearn.preprocessing import LabelEncoder
# Identify object columns
object_columns = df.select_dtypes(include=['object']).columns

# Create a LabelEncoder instance
labelencoder = LabelEncoder()

# Encode object columns
for column in object_columns:
    if column!= "image":
        df[column] = labelencoder.fit_transform(df[column])
df