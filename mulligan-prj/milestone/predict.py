sample = 481
prediction = classifier.predict(np.reshape(X_test[sample], (1,64,
    64,3)))
print(prediction)
print(np.argmax(prediction)+1)
print(y_test[sample])
print(np.argmax(y_test[sample])+1)


x = np.reshape(X_test[sample,:,:,1],(64,64))
io.imshow(x)