import numpy.typing as npt
import numpy as np

num_points = 1000

x: npt.NDArray[np.int_] = np.arange(0, num_points)

def dominant_frequency_likelihood(y: npt.NDArray[np.int_]) -> float:
    # Perform Fourier Transform on the dataset
    fft_values = np.fft.fft(y)

    # Get the absolute values (magnitudes) of the FFT
    magnitudes = np.abs(fft_values)

    # Get the frequencies corresponding to FFT values
    frequencies = np.fft.fftfreq(len(y), d=1)  # d=1 means sample spacing is 1

    # Find the index of the maximum magnitude, excluding the zero-frequency component
    dominant_frequency_index = np.argmax(magnitudes[1:]) + 1
    dominant_magnitude = magnitudes[dominant_frequency_index]

    # Calculate the average magnitude, excluding the zero-frequency component
    average_magnitude = np.mean(magnitudes[1:])

    # Calculate the ratio of the dominant magnitude to the average magnitude
    likelihood = dominant_magnitude / average_magnitude

    return likelihood

def test_mod3():
    y = x % 3
    likelihood = dominant_frequency_likelihood(y)
    print(f"Likelihood of periodic patterns in mod3: {likelihood}")

def test_random():
    y: npt.NDArray[np.int_] = np.random.randint(0, 100, size=num_points) # type: ignore
    likelihood = dominant_frequency_likelihood(y)
    print(f"Likelihood of periodic patterns in random: {likelihood}")

# Run the tests
test_mod3()
test_random()
