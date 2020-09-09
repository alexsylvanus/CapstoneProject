import numpy as np
import soundfile
from math import floor
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from sys import argv

def main(d, a, s=''):
    if not s:
        # create beep sound with pitch set by distance
        sampleRate = 44100
        frequency = 1500*(1-d/100)
        length = .5
        t = np.linspace(0, length, sampleRate * length)  
        sound = np.sin(frequency * 2 * np.pi * t)
        soundfile.write('test.wav', sound, sampleRate, subtype='PCM_16')
        audio = AudioSegment.from_wav("test.wav")
    else:
        language = 'en'
        sound = gTTS(text = s, lang = language, slow = False)
        sound.save("test.mp3")
        audio = AudioSegment.from_mp3("test.mp3")
        audio = audio + floor((50 - d)/2)

    out = audio.pan(a/100)
    play(out)

if __name__ == '__main__':
    if len(argv) < 3:
        print 'too few arguments.'
    if len(argv) == 3:
        main(argv[1], argv[2])
    if len(argv) == 4:
        main(argv[1], argv[2], argv[3])
    else:
        print 'too many arguments.'
    
