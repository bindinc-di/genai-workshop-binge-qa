"""
The script transforms binge app feed (example gs://bdc_binge_app/test/2024-10-24/popcorn.ndjson)
into a text document suitable for semantic search
It combines the title, intro and content and metadata (cast, availability etc) into a topic.
If an optional imdb plots file provided adds plots of the matching IMDB ids
"""

from argparse import ArgumentParser
import json
import csv
import os
from datetime import datetime


CURRENT_YEAR = datetime.now().year

# import numpy as np
# import pandas as pd


def safeget(dic: dict, path: str | list):
    """Safely get a value from a nested dictionary using a path."""
    if isinstance(path, str):
        path = path.split(".")

    for key in path:
        try:
            dic = dic[key]
        except KeyError:
            return None
    return dic


def desc_genres(x):
    if not x or not isinstance(x, list):
        return ""
    return "Genres: " + ", ".join(x)


def decs_director(x):
    return "Director: " + x.strip() if x else ""


def desc_year(x):
    return (
        "Productiejaar: " + str(x)
        if x and str(x).isdigit() and 1900 < int(x) < CURRENT_YEAR
        else ""
    )


def desc_actors(x):
    if not x or not isinstance(x, list):
        return ""
    return "Acteurs: " + ", ".join(x)


def desc_synopsis(x):
    if not x:
        return ""
    return "Synopsis: " + x.strip()


def desc_availability(item_type, suppliers):
    translate_item_type_NL = {"series": "serie", "movie": "film"}
    res = ""
    if suppliers and isinstance(suppliers, list):
        res += (
            "Deze "
            + translate_item_type_NL.get(item_type, "programma")
            + " is te zien op "
            + "en ".join([s["user_friendly_name"] for s in suppliers])
            + "."
        )
    return res


def desc_main_item(x):
    item_type = x["item_type"]
    res = [
        desc_synopsis(x.get("long_description") or x.get("short_description")),
        decs_director(x["director"]),
        desc_year(x["year"]),
        desc_actors(x["actor"]),
        desc_genres(x["genres"]),
        desc_availability(item_type, x["supplier"]),
    ]

    res = [r for r in res if r and r.strip() != ""]

    if len(res) > 0:
        return "\n".join(res)
    return ""


def remove_html_tags(text):
    """Remove html tags from a string"""
    import re

    clean = re.compile("<.*?>")
    return re.sub(clean, "", text).strip()


def get_text_from_blocks(blocks):
    res = ""

    for block in blocks:
        if block["type"] == "paragraph":
            if text := safeget(block, "data.text"):
                res += str(text) + "\n"
        elif block["type"] == "header":
            if text := safeget(block, "data.text"):
                res += "### " + str(text) + "\n"
        elif block["type"] == "raw":
            if text := safeget(block, "data.html"):
                res += remove_html_tags(text) + "\n"
        else:
            print("Unknown block type:", block["type"])

    return res


def rec_2_text(x: dict):
    res = "\n".join(
        [
            x["title"],
            x["intro"],
            x["content"],
        ]
    )
    if "main_item" in x and isinstance(x["main_item"], dict):
        res += "\n" + desc_main_item(x["main_item"])

    if (
        "json_content" in x
        and isinstance(x["json_content"], dict)
        and "blocks" in x["json_content"]
        and isinstance(x["json_content"]["blocks"], list)
    ):
        res += "\n" + get_text_from_blocks(x["json_content"]["blocks"])

    return res


def get_bindinc_api_uri(rec):
    if "bindinc_api" in rec:
        return rec["bindinc_api"][0]["uri"].split("/")[-1]
    return ""


def process_file(file_name: str, imdb_file_name: str, out_file_format: str) -> None:

    assert out_file_format in ["ndjson", "txt", "csv"], "Unexpected output file format"

    out_file_name = os.path.splitext(file_name)[0] + ".processed." + out_file_format
    print("Output file:", out_file_name)

    imdb_plots = {}
    if imdb_file_name and os.path.isfile(imdb_file_name):
        print("Enriching with IMDB file:", imdb_file_name)
        with open(imdb_file_name, "r") as f:
            for r in f:
                rec = json.loads(r)
                imdb_plots[rec["imdbId"]] = (
                    rec["plotSummary"] + "\n" + rec["plotSynopsis"]
                ).strip(" \n")

    with open(file_name, "r") as f, open(out_file_name, "w") as w:
        if out_file_format == "csv":
            cw = csv.writer(w, delimiter="\t")
            cw.writerow(["id", "bindinc_id", "page_content"])
        for i, line in enumerate(f):
            rec = json.loads(line)
            page_content = rec_2_text(rec)

            main_item = rec.get("main_item") or {}
            streaming_identifier = main_item.get("streaming_identifier", "").strip(" -")
            
            imdb_id_attr_name = "imdb_url_series" if main_item.get("item_type") == "serie" else "imdb_url_program"

            imdb_id = main_item.get(imdb_id_attr_name)

            if imdb_id in imdb_plots:
                page_content += "\n\nSynopsis: " + imdb_plots[imdb_id]

            id_ = rec["id"]
            bindinc_id = get_bindinc_api_uri(rec)
            title = rec["title"]
            image = rec["image_popcorn"]
            if out_file_format == "ndjson":

                w.write(
                    json.dumps(
                        {
                            "page_content": page_content,
                            "metadata": {
                                "title": title,
                                "id": id_,
                                "binge_id": streaming_identifier,
                                "image_urls": image,
                                "imdb_id": imdb_id
                            },
                        }
                    )
                    + "\n"
                )
            elif out_file_format == "txt":
                w.write(page_content + "\n")
            elif out_file_format == "csv":
                cw.writerow([id_, bindinc_id, page_content])


def main():
    parser = ArgumentParser()
    parser.add_argument("file_name")
    parser.add_argument(
        "--imdb-file-name", default="", help="Additional imdb file with plots"
    )
    parser.add_argument(
        "--output-format",
        default="ndjson",
        choices=["ndjson", "txt", "csv"],
        help="Output file format",
    )

    args = parser.parse_args()

    print("Processing file:", args.file_name)

    # process_file(args.file_name, args.output_format)
    process_file(args.file_name, args.imdb_file_name, args.output_format)

    print("Done.")


if __name__ == "__main__":
    main()
