import os
import utils.parser
import datasets.reader

args = utils.parser.args()

OUTPUT_FOLDER = args.save
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

images, labels = datasets.reader.get_data(args.images, args.labels)
'''
print(len(images), len(labels))

print(images[0].shape, images[0].dtype)
print(labels[0].shape, labels[0].dtype)
print(images[1].shape, images[1].dtype)
print(labels[1].shape, labels[1].dtype)
'''
