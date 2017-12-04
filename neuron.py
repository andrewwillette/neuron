from numpy import exp, array, random, dot, rint
import urllib
import scipy.io.wavfile
import pydub

#a temp folder for downloads
temp_folder="/Users/andrewwillette/Documents/githubProjects/simpleNeuralNetwork/neuron/AudioSamples/"

#read mp3 file
mp3SampleOne = pydub.AudioSegment.from_mp3(temp_folder+"LetItGo.mp3")
mp3SampleTwo = pydub.AudioSegment.from_mp3(temp_folder+"DirtyWork.mp3")
mp3SampleThree = pydub.AudioSegment.from_mp3(temp_folder+"PlasticLove.mp3")
mp3SampleFour = pydub.AudioSegment.from_mp3(temp_folder+"beibsInTrap.mp3")

#convert to wav
mp3SampleOne.export(temp_folder+"LetItGo.wav", format="wav")
mp3SampleTwo.export(temp_folder+"DirtyWork.wav", format="wav")
mp3SampleThree.export(temp_folder+"PlasticLove.wav", format="wav")
mp3SampleFour.export(temp_folder+"beibsInTrap.wav", format="wav")

#read wav file
rateOne,audDataOne=scipy.io.wavfile.read(temp_folder+"LetItGo.wav")
rateTwo,audDataTwo=scipy.io.wavfile.read(temp_folder+"DirtyWork.wav")
rateThree,audDataThree=scipy.io.wavfile.read(temp_folder+"PlasticLove.wav")
rateFour,audDataFour=scipy.io.wavfile.read(temp_folder+"beibsInTrap.wav")

print(rateOne)
#print(audData)
#for p in audData: print p[0]
signalOne = []
signalTwo = []
signalThree = []
signalFour = []
for x in range(5000, 8000) : 
	#print audData[x][0]
	signalOne.append(audDataOne[x][0])
	signalTwo.append(audDataTwo[x][0])
	signalThree.append(audDataThree[x][0])
	signalFour.append(audDataFour[x][0])



#training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_set_inputs = array([signalOne, signalTwo, signalFour])


training_set_outputs = array([[1,0,1]]).T

random.seed(1)

synaptic_weights = 2 * random.random((3000, 1)) - 1

for iteration in xrange(10000):

    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
    
    synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))

print 1 / (1 + exp(-(dot(array(signalThree), synaptic_weights))))