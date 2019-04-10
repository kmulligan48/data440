print(X.shape)

x = np.reshape(X[4,:,:,1],(64,64))
print(x.shape)
plt.imshow(x)

#print(labels)