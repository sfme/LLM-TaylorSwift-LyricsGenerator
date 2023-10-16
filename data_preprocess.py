import pandas as pd
import os
import re
import argparse


def get_lyrics(row):
    base, album_name = row["Album_DirPath"].rsplit("/", 1)

    path = (
        base
        + "/"
        + (
            album_name
            + "/"
            + re.sub("[^a-zA-Z0-9_]", "_", "".join(row["Tracks"].split()))
            + ".txt"
        )
        .replace("_s", "s")
        .replace("the_ladieslunching_chapter", "theladieslunchingchapter")
        .replace(
            "Carolina_FromTheMotionPicture_WhereTheCrawdadsSing__",
            "Carolina_FromTheMotionPictureWhereTheCrawdadsSing_",
        )
        .replace(
            "evermore_the_foreveristhesweetestcon_chapter",
            "evermore_theforeveristhesweetestconchapter",
        )
        .replace("willow_90strendremix_.txt", "/willow_90_strendremix_.txt")
        .replace("MeetMeAtMidnight.txt", "Meetmeatmidnight.txt")
        .replace("DeathByAThousandCuts.txt", "DeathbyaThousandCuts.txt")
        .replace("ItsNiceToHaveAFriend.txt", "ItsNicetoHaveaFriend.txt")
    )

    try:
        with open(path, "r") as f:
            txt = f.read()
        return txt

    except Exception as e:
        print(path)
        pass


def lyrics_preprocess(lyrics_txt):
    # removes header until "Lyrics" word, and removes end of lyrics "Embed" word
    lyrics_txt = lyrics_txt.split("Lyrics", 1)[1][: -len("Embed")]

    # removes number at end if exists (some lyrics have it, some don't)
    return re.sub(r"\d+$", "", lyrics_txt)


def preprocess_data(path_data, path_save):
    # paths to data files for lyrics and related albums
    path_tracks = os.path.join(path_data, "Tabular/")
    path_album = os.path.join(path_data, "Albums/")

    # Get albums data in pandas DataFrame form
    album_df = pd.read_csv(os.path.join(path_data, "Albums.csv"))
    album_df.drop("Unnamed: 0", axis=1, inplace=True)

    # Get tracks data in pandas DataFrame form
    album_df["Track_path"] = album_df["Albums"].apply(
        lambda x: os.path.join(
            path_tracks, re.sub("[^a-zA-Z0-9_]", "_", "".join(x.split())) + ".csv"
        )
    )

    # Get entire dataset in a DataFrame format, for use in train / test
    dataset_df = pd.DataFrame()
    for i in range(len(album_df)):
        try:
            tmp_track_df = pd.read_csv(album_df.loc[i]["Track_path"])
            tmp_track_df["Album_ID"] = album_df.loc[i]["ID"]
            tmp_track_df["Album"] = album_df.loc[i]["Albums"]
            tmp_track_df["Album_DirPath"] = os.path.join(
                path_album,
                re.sub(
                    "[^a-zA-Z0-9_]", "_", "".join(album_df.loc[i]["Albums"].split())
                ),
            )
            dataset_df = pd.concat([dataset_df, tmp_track_df])

        except Exception as e:
            print("there is no track for this album")
            pass

    # reset index of dataset DataFrame, needs it due to concatenation above
    dataset_df.reset_index(drop=True, inplace=True)
    # drop irrelevant column
    dataset_df.drop("Unnamed: 0", axis=1, inplace=True)

    # get all the lyrics from the tracks that exist, insert into dataset df
    dataset_df["lyrics"] = dataset_df.apply(get_lyrics, axis=1)

    # remove any empties from the full dataset
    dataset_df.dropna(inplace=True)
    dataset_df.reset_index(drop=True, inplace=True)

    # process lyrics to remove any starting or ending tokens not needed
    dataset_df["lyrics"] = dataset_df.lyrics.apply(lyrics_preprocess)

    # saving the current full dataset DataFrame to file (.json)
    dataset_df.to_json(
        os.path.join(path_save, "postproc_dataset_ts_lyrics.json"),
        orient="split",
        compression="infer",
        index="true",
    )


def main():
    parser = argparse.ArgumentParser(description="pre-process taylor swift lyrics")

    parser.add_argument("--path-data", type=str, help="dataset filepath")
    parser.add_argument(
        "--path-save", type=str, help="save final pre-processed dataset filepath"
    )

    args = parser.parse_args()

    preprocess_data(args.path_data, args.path_save)


if __name__ == "__main__":
    main()
