import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import time
from collections import deque

def WithPause():
    plt.ion()

    fig, ax = plt.subplots()

    x = []
    y = []

    line, = ax.plot(x, y)

    counter = 0

    while True:
        value = random.randint(0, 100)

        x.append(counter)
        y.append(value)

        line.set_data(x, y)

        ax.relim()
        ax.autoscale_view()

        plt.pause(0.1)

        counter += 1

def WithFuncAnim():
    # Store the last 100 data points
    x_data = deque(maxlen=100)
    y_data = deque(maxlen=100)

    fig, ax = plt.subplots()

    line, = ax.plot([], [], 'b-', linewidth=2)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Live Data Plot")
    ax.grid(True)

    counter = 0

    def update(frame):
        global counter

        # Generate new data
        value = random.randint(0, 100)

        x_data.append(counter)
        y_data.append(value)

        line.set_data(x_data, y_data)

        counter += 1

        # Automatically adjust x-axis
        if counter > 100:
            ax.set_xlim(counter - 100, counter)

            return line,

        ani = animation.FuncAnimation(
            fig,
            update,
            interval=100,  # Update every 100 ms
            blit=False,
            cache_frame_data=False
        )

WithFuncAnim()
plt.show()
