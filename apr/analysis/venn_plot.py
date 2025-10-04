#!/usr/bin/env python3
# pip install venn matplotlib

import matplotlib.pyplot as plt
from venn import venn  # supports up to 6 sets

# Generates and saves a Venn diagram of overlap among apr_id sets for RAMP, LANTERN, ChatRepair, and Self-Planning, also printing each set’s size.

# === apr_id sets (deduped) ===
CHATREPAIR = {
    "8c74e7e7ec387343ac5d678b1d5c5b48",
    "358598c686419603c1953ff11a181904",
    "4fb6ec4039d364e73bc614c78329db03",
    "569f76b34a8da20ca0e8051e0b5fa29f",
    "87d1c3cf50718b554eedfedc7e64bf22",
    "5f5a0651c1811fd31ac3b16bd41b5b1a",
}

LANTERN = {
    "8c74e7e7ec387343ac5d678b1d5c5b48",
    "4fb6ec4039d364e73bc614c78329db03",
    "e34c4dc681c216ab35dcf707a6560517",
    "7be3bcfc27dd8686d28176a5cb742e0f",
    "c741cb93a321a13d10d1d17707bb43c0",
    "9eae7c8a50d0cc25a94ff7c82a1be1d9",
    "fbb49dec1c2f08a2b02ea046bb552cd7",
    "1746e8b6147adfa45926cc98f18d4bdc",
    "0765c2555150aedc476e30b9176b33cf",
    "cb59b37fb345949c7b725445922565d2",
    "0b5f31db6fcc73db5582842d8f4426ea",
    "c08ed1dff9b3897f7617fae0db9f7082",
    "cfefcb65327bcd4b5c80e5a2cabae117",
    "d139c1a9df5ef7661cb6a7a13d10bad5",
    "87d1c3cf50718b554eedfedc7e64bf22",
    "358598c686419603c1953ff11a181904",
    "569f76b34a8da20ca0e8051e0b5fa29f",
    "af89fc60e0ec22f0647d409271774550",
    "4ccfd1749548833794283b10dd789230",
    "f87d3926724274e5d18cd71e223eebc5",
    "2e722ee6410f288186ef2b61f2afcbfd",
}

SELFPLANNING = {
"8c74e7e7ec387343ac5d678b1d5c5b48",
"cb59b37fb345949c7b725445922565d2",
"2e722ee6410f288186ef2b61f2afcbfd",
"cfefcb65327bcd4b5c80e5a2cabae117",
"9eae7c8a50d0cc25a94ff7c82a1be1d9",
"87d1c3cf50718b554eedfedc7e64bf22",
"d139c1a9df5ef7661cb6a7a13d10bad5",
"4fb6ec4039d364e73bc614c78329db03",
"fbb49dec1c2f08a2b02ea046bb552cd7",
"1746e8b6147adfa45926cc98f18d4bdc",
"569f76b34a8da20ca0e8051e0b5fa29f",
"cd60f5145685a6cb7ce0d8773a4fe64b",
"0765c2555150aedc476e30b9176b33cf",
"4ccfd1749548833794283b10dd789230",
}

RAMP = {
    "2e722ee6410f288186ef2b61f2afcbfd",
    "cb59b37fb345949c7b725445922565d2",
    "c741cb93a321a13d10d1d17707bb43c0",
    "8c74e7e7ec387343ac5d678b1d5c5b48",
    "cfefcb65327bcd4b5c80e5a2cabae117",
    "358598c686419603c1953ff11a181904",
    "9eae7c8a50d0cc25a94ff7c82a1be1d9",
    "d139c1a9df5ef7661cb6a7a13d10bad5",
    "87d1c3cf50718b554eedfedc7e64bf22",
    "4fb6ec4039d364e73bc614c78329db03",
    "569f76b34a8da20ca0e8051e0b5fa29f",
    "5f5a0651c1811fd31ac3b16bd41b5b1a",
    "fbb49dec1c2f08a2b02ea046bb552cd7",
    "1746e8b6147adfa45926cc98f18d4bdc",
    "60021813eb8e614db64f00042bbe867f",
    "e34c4dc681c216ab35dcf707a6560517",
    "efa077c51888131f3cecb9d011c30b72",
    "0765c2555150aedc476e30b9176b33cf",
    "af89fc60e0ec22f0647d409271774550",
    "7be3bcfc27dd8686d28176a5cb742e0f",
    "c08ed1dff9b3897f7617fae0db9f7082",
    "4ccfd1749548833794283b10dd789230",
    "f87d3926724274e5d18cd71e223eebc5",
}

DATA = {
    "ChatRepair": CHATREPAIR,
    "LANTERN": LANTERN,
    "Self-Planning": SELFPLANNING,
    "RAMP": RAMP,
}

def main():
    # quick summary
    print("Set sizes (apr_id):")
    for name, s in DATA.items():
        print(f"  {name}: {len(s)}")

    # plot venn
    plt.figure(figsize=(5, 1), dpi=500)
    ax = venn(DATA, legend_loc="center right")

    # Get the legend and shift it a bit
    leg = ax.get_legend()
    leg.set_bbox_to_anchor((1.25, 0.5))
    # plt.title("Solved overlaps by apr_id")
    plt.tight_layout()
    plt.savefig("venn_apr_ids.png")
    plt.show()

if __name__ == "__main__":
    main()
