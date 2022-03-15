import matplotlib.pyplot as plt
import numpy
import json
from pprint import pprint
import sys
import argparse
from scipy.interpolate import make_interp_spline, BSpline
import numpy as np
import os
import re


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def main(args):
	parser = argparse.ArgumentParser(description='Pommerman agent plot script')
	parser.add_argument('-d', '--dir', type=str, dest="dir",
						help=f"Name of directory containing json logs")
	parser.add_argument('-x', '--xlabel', type=str, dest="xlabel", default="Iterations",
						help=f"Label for x axis")
	parser.add_argument('-y', '--ylabel', type=str, dest="ylabel", default="Win Ratio",
						help=f"Label for y axis")

	args = parser.parse_args(args[1:])

	dir = args.dir
	for file in os.listdir(dir):

		if not file.endswith(".json"):
			continue

		print(file)
		tag = re.search('run-(.*?)-tag', file).group(1)

		with open(dir + file) as json_file:
			data = json.load(json_file)

		points = list(zip(*data))
		x, y = np.array(points[1]), np.array(points[2])

		y = smooth(y, 0.98)

		plt.plot(x[:150], y[:150], label=tag)


	plt.legend()
	plt.xlabel(args.xlabel)
	plt.ylabel(args.ylabel)
	plt.show()

if __name__ == '__main__':
	main(sys.argv)
