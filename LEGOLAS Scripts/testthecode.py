import rpyc
from ImageAssessment import *


model, data_transform = load_model('model_10.pt')

processed_image = preprocess_image('spotD.jpeg')

image = data_transform(processed_image)
img_list = [image]

pred = make_prediction(model, img_list, 0.5)
pred_array = pred[0]['boxes'].detach().numpy()

coords, x_coords, y_coords = convert_to_coords(pred_array)

v_group, h_group, v_line, h_line = make_grid_lines(coords)

plot_grid(x_coords, y_coords, v_line, h_line, processed_image)