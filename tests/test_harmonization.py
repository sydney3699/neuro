"""Unit tests for the GW20->common-class harmonization mapping.

These lock the persistence-critical mapping (the committed GW34 results depend on
it) against accidental change.
"""
from neurospatial.harmonization import GW20_H2_TO_COMMON, COMMON


def test_common_has_eight_classes():
    assert len(COMMON) == 8
    assert "Other-progenitor" in COMMON


def test_every_mapped_value_is_a_common_class():
    assert set(GW20_H2_TO_COMMON.values()) <= set(COMMON)


def test_en_l2_maps_to_upper_layer():
    assert GW20_H2_TO_COMMON["EN-L2"] == "EN-IT-UL"


def test_en_it_layer_split():
    # upper vs deep layer EN-IT split (the persistent V1/V2 signal)
    assert GW20_H2_TO_COMMON["EN-IT-L2/3"] == "EN-IT-UL"
    assert GW20_H2_TO_COMMON["EN-IT-L4"] == "EN-IT-UL"
    assert GW20_H2_TO_COMMON["EN-IT-L4/5"] == "EN-IT-DL"
    assert GW20_H2_TO_COMMON["EN-IT-L6"] == "EN-IT-DL"


def test_progenitors_bucketed_not_dropped():
    for t in ("RG1", "oRG1", "vRG-late", "IPC-SVZ-1", "EN-IZ-1", "EN-oSVZ-1"):
        assert GW20_H2_TO_COMMON[t] == "Other-progenitor"


def test_subplate_en_et_kept_as_en_et():
    for t in ("EN-ET-SP", "EN-ET-SP-P", "EN-ET-SP-early", "EN-ET-L5/6"):
        assert GW20_H2_TO_COMMON[t] == "EN-ET"


def test_representative_gw20_h2_all_mapped():
    # a representative slice of the real GW20 commot H2 vocabulary all resolves
    seen = ["EN-IT-L4", "EN-IT-L6", "EN-ET-SP", "IN-MGE", "EC", "Astro-1", "OPC", "RG1"]
    for t in seen:
        assert GW20_H2_TO_COMMON[t] in COMMON
