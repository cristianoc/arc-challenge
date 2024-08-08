import numpy.typing as npt
import numpy as np
import matplotlib.pyplot as plt

num_points = 1000

x: npt.NDArray[np.int_] = np.arange(0, num_points)

# Compute x % 3 for the larger dataset
y = x % 3

def dominant_frequency_and_period(y: npt.NDArray[np.int_], s: str) -> None:
    # Perform Fourier Transform on the larger dataset
    fft_values_large = np.fft.fft(y)

    # Get the absolute values (magnitudes) of the FFT for the larger dataset
    magnitudes_large = np.abs(fft_values_large)

    # Get the frequencies corresponding to FFT values for the larger dataset
    frequencies_large = np.fft.fftfreq(len(y), d=1)  # d=1 means sample spacing is 1

    # Find the index of the maximum magnitude, excluding the zero-frequency component for the larger dataset
    dominant_frequency_index_large = np.argmax(magnitudes_large[1:]) + 1
    dominant_frequency_large = frequencies_large[dominant_frequency_index_large]

    # Calculate the period for the larger dataset
    dominant_period_large = 1 / dominant_frequency_large

    # Plot the FFT magnitudes
    plt.figure()
    plt.plot(frequencies_large, magnitudes_large)
    plt.title(f"FFT Magnitudes for {s}")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.show()

    print(f"Results for {s}")
    print(f"Dominant Frequency: {dominant_frequency_large}")
    print(f"Dominant Period: {dominant_period_large}")

def test_mod3():
    y = x % 3
    dominant_frequency_and_period(y, "mod3")

def test_random():
    y: npt.NDArray[np.int_] = np.random.randint(0, 100, size=num_points) # type: ignore
    dominant_frequency_and_period(y, "random")
