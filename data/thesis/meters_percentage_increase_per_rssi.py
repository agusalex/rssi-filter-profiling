import numpy as np
import matplotlib.pyplot as plt

# Define the RSSI calculation function with the given A and N
def calculate_rssi(d, A=43, N=31):
    return -A - N * np.log10(d)

# Function to calculate the change in distance for a 1 dBm increase in RSSI
def distance_change_per_rssi_increase_custom(d, A=43, N=31):
    rssi = calculate_rssi(d, A, N)  # Calculate RSSI for the given distance
    d_plus_1_rssi = np.power(10, -(rssi + 1 + A) / N)  # Calculate distance for RSSI + 1 with custom A and N
    return np.abs(d_plus_1_rssi - d)  # Return the absolute change in distance

# Generate a range of distances
distances = np.linspace(1, 100, 500)  # From 1 to 100 meters

# Calculate the change in distance for a 1 dBm increase in RSSI over the range of distances
distance_change_custom = np.array([distance_change_per_rssi_increase_custom(d) for d in distances])

plt.figure(figsize=(10, 6))
plt.plot(distances, distance_change_custom, label='Distance Change per 1 dBm RSSI Increase', color='orange')
plt.xlabel('Distance (m)')
plt.ylabel('Distance Change (m)')
plt.title('Distance Change per 1 dBm Increase in RSSI with A=43, N=31')
plt.xticks(np.arange(0, 16, 5).put(np.arange(16, range(distances), 10)))  # Set x-axis to mark every 5 meters
plt.axvline(x=10, color='red', linestyle='--', label='Increased Variability Threshold')
plt.text(10, max(distance_change_custom) / 2, ' Increased Variability Zone', color='red', va='bottom')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
