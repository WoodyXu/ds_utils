# -*- encoding: utf-8 -*-

"""
Author: Woody
Descrption:
    This is a module about flight features
"""
import json
import os

airlines_conf = None

def __load_airlines_configuration():
    """
    internal function
    """

    global airlines_conf
    conf_file =  "%s/airlines.json" % os.path.dirname(os.path.abspath(__file__))
    airlines_conf = json.load(open(conf_file, "r"))

def airline_alliance(iata_code):
    """
    Parameters:
        iata_code: airline's IATA Code, such as CA for AirChina, case insensitive
    Returns:
        "sa" for StarAlliance
        "st" for SkyTeam
        "ow" for OneWorld
        "" for bad input or non alliance
    """

    global airlines_conf

    result_codes = {"sa": "staralliance", "st": "skyteam", "ow": "oneworld"}

    iata_code = iata_code.upper()
    for code in result_codes:
        airline_list = airlines_conf[result_codes[code]]
        if iata_code in airline_list:
            return code
    return ""

def is_low_cost_airline(iata_code):
    """
    Parameters:
        iata_code: airline's IATA Code, such as CA for AirChina, case insensitive
    Returns:
        True means low cost carrier, False means full service airline or bad input
    """

    global airlines_conf

    return iata_code.upper() in airlines_conf["lcc"]

def unit_test():
    __load_airlines_configuration()

    # airline_alliance
    assert airline_alliance("ca") == "sa"
    assert airline_alliance("MU") == "st"
    assert airline_alliance("Ba") == "ow"
    assert airline_alliance("EK") == ""
    assert airline_alliance("abcdefg") == ""

    # is_low_cost_airline
    assert is_low_cost_airline("ca") is False
    assert is_low_cost_airline("9c") is True

if __name__ == "__main__":
    unit_test()
