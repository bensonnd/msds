list1 = [1, 2, 3, 4, 5, 6]
list2 = [9, 10, 11, 12, 13, 14]

models = zip(list1, list2)

for model in models:
    print(model[1])
    a = 1
    # print(f"{model[1]}__+")

l1, l2 = zip(*models)
# print(f"{l1}__+")
# print(f"{l2}^^_")


a = 1


dict_ = {
    "key1": [("X_train", "y_train", "X_test", "y_test"), "clf"],
    "key2": [("X_train", "y_train", "X_test", "y_test"), "clf"],
}

