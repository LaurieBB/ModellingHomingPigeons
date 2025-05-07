import csv

from bs4 import BeautifulSoup
from requests_html import HTMLSession

session = HTMLSession()

# Gets the correct url to find the geomagnetic vectors
def get_url(latitude, longitude, altitude):
    url = f"https://geomag.bgs.ac.uk/web_service/GMModels/wmmhr/2025?latitude={latitude}&longitude={longitude}&altitude={altitude}&date=2025-01-01&format=xml"

    return url

# Sort the output and get rid of the tags, returning only the values.
def clear_output(output):
    # Values are saved in the same order as presented on the website https://geomag.bgs.ac.uk/data_service/models_compass/wmm_calc.html
    # These are output and used to create the map, to show geomagnetic strength.
    values = []
    output = output.splitlines()
    output = [x.replace(" ", "") for x in output]
    output = output[output.index("<field-value>"):output.index("<secular-variation>")]

    values.append(float(output[output.index("</declination>") - 1]))
    values.append(float(output[output.index("</inclination>") - 1]))
    values.append(float(output[output.index("</north-intensity>") - 1]))
    values.append(float(output[output.index("</east-intensity>") - 1]))
    values.append(float(output[output.index("</horizontal-intensity>") - 1]))
    values.append(float(output[output.index("</vertical-intensity>") - 1]))
    values.append(float(output[output.index("</total-intensity>") - 1]))

    return values

# Get all the geomagnetic values for the coordinates in the map
def get_mag_values():
    mag_values = []
    with open("data/Coordinates.csv", "r") as f:
        coordinates = csv.reader(f)
        for lines in coordinates:
            latitude = str(lines[0]).replace(" ", "")
            longitude = str(lines[1]).replace(" ", "")
            altitude = 0.18288
            #TODO This altitude is 600ft in km, as required by the website. Need to come back and check it is an accurate value
            # and also write up that I chose to use this value.

            url = get_url(latitude, longitude, altitude)
            res = session.get(url)
            soup = BeautifulSoup(res.content, features="xml")
            output = soup.prettify()
            mag_values.append(clear_output(output))

    return mag_values

# Due to the order required by the website to input coordinates the values are in the incorrect order and must be sorted.
# Therefore, this returns an array of the values in the matching order to the grids in the map
def reorder_vals(values):
    final_values = []
    temp_lists = {}
    for x in range(0, 10):
        temp_lists[x] = values[x*10:10*(x+1)]

    for key in temp_lists.keys():
        if key % 2 == 0:
            final_values.append(temp_lists[key])
        else:
            reversed = temp_lists[key][::-1]
            final_values.append(reversed)

    return final_values


def generate_map():
    mag_values = get_mag_values()
    return reorder_vals(mag_values)

