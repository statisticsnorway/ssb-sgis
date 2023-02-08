import geopandas as gpd
import numpy as np
from dataclasses import dataclass
from copy import copy, deepcopy

from .od_cost_matrix import od_cost_matrix
from .shortest_path import shortest_path
from .service_area import service_area
from .nettverk import lag_node_ids


# årene det ligger tilrettelagte vegnettverk på Dapla.
# hvis man vil bruke et annet nettverk, kan man kjøre det gjennom networkz.nettverk.lag_nettverk.
NYESTE_AAR = 2022
ELDSTE_AAR = 2019


"""
NETTVERKSSTI_BIL = f"C:/Users/ort/OneDrive - Statistisk sentralbyrå/data/klargjorte_nettverk/nettverk_bil_{NYESTE_AAR}.parquet"
NETTVERKSSTI_SYKKELFOT = f"C:/Users/ort/OneDrive - Statistisk sentralbyrå/data/klargjorte_nettverk/nettverk_sykkelfot_{NYESTE_AAR}.parquet"
KOMMUNESTI = (f"C:/Users/ort/OneDrive - Statistisk sentralbyrå/data/Basisdata_0000_Norge_25833_Kommuner_FGDB.gdb", "kommune")
"""

# dapla-filstier
NETTVERKSSTI_BIL = f"ssb-prod-dapla-felles-data-delt/GIS/Vegnett/Klargjorte_nettverk/nettverk_bil_{NYESTE_AAR}.parquet"
NETTVERKSSTI_SYKKELFOT = f"ssb-prod-dapla-felles-data-delt/GIS/Vegnett/Klargjorte_nettverk/nettverk_sykkelfot_{NYESTE_AAR}.parquet"
KOMMUNESTI = f"ssb-prod-dapla-felles-data-delt/GIS/{NYESTE_AAR}/ABAS_kommune_flate_{NYESTE_AAR}.parquet"


""" Først regler for kjøretøyene. I hver sin class for å gjøre det litt mer oversiktlig. 
Class-ene har ingen effekt utover å definere standardreglene for kjøretøyet som velges i Graf-class-en.
Reglene (parametrene) kan settes i Graf(). """


@dataclass
class ReglerFot:
    fart: int = 5
    prosent_straff_stigning: int = 5
    max_aadt: int = None
    max_fartsgrense: int = None
    forbudte_vegtyper: tuple = ("Sykkelfelt", "Sykkelveg")
    nettverkssti: str = NETTVERKSSTI_SYKKELFOT


@dataclass
class ReglerSykkel:
    fart: int = 20
    prosent_straff_stigning: int = 23
    max_aadt: int = None
    max_fartsgrense: int = None
    forbudte_vegtyper: tuple = (
        "Fortau",
        "Sti",
        "Stor sti",
        "Gangveg",
        "Gangfelt",
    )  # det er jo egentlig ikke forbudt å sykle her...
    nettverkssti: str = NETTVERKSSTI_SYKKELFOT


@dataclass
class ReglerBil:
    turn_restrictions: bool = False  # svingforbud. midlr false
    sperring: str = "ERFKPS"  # hvilke vegkategorier hvor vegbommer skal være til hinder. Hvis sperring er None, er alle bommer lov. Hvis sperring=='ERFKS', er det lov å kjøre gjennom private bommer.
    nettverkssti: str = NETTVERKSSTI_BIL


class Graf:
    """
    Class som inneholder vegnettet, nodene og generelle regler for hvordan nettverksanalysen skal gjennomføres.
    Regler knyttet til kjøretøyet går via ReglerSykkel, ReglerFot og ReglerBil, men parametrene godtas her.
    """

    def __init__(
        self,
        # disse handler om hvilket nettverk som skal leses inn, og hvilket område som skal beholdes
        aar=NYESTE_AAR,
        *,
        kjoretoy="bil",
        kommuner=None,
        # hvis man vil bruke et annet nettverk
        nettverk: gpd.GeoDataFrame = None,
        # generelle regler for nettverksanalysen
        directed=True,
        kostnad="minutter",
        fjern_isolerte=True,
        search_tolerance=1000,
        dist_faktor=10,
        kost_til_nodene: int = 5,
        # regler knyttet til kjøretøyet (parametrene i kjøretøy-class-ene)
        **qwargs,
    ):
        self._aar = aar
        self.nettverk = nettverk

        # sørg for at kommunene er i liste og har 4 bokstaver
        if kommuner:
            if isinstance(kommuner, (str, int, float)):
                self._kommuner = [str(int(kommuner)).zfill(4)]
            else:
                self._kommuner = [str(int(k)).zfill(4) for k in kommuner]
        else:
            self._kommuner = None

        # hent ut reglene for kjøretøyet
        kjoretoy = kjoretoy.lower()
        if "bil" in kjoretoy or "car" in kjoretoy or "auto" in kjoretoy:
            self._kjoretoy = "bil"
            self.regler = ReglerBil(**qwargs)
        elif "sykkel" in kjoretoy or "bike" in kjoretoy or "bicyc" in kjoretoy:
            self._kjoretoy = "sykkel"
            self.regler = ReglerSykkel(**qwargs)
        elif "fot" in kjoretoy or "foot" in kjoretoy:
            self._kjoretoy = "fot"
            self.regler = ReglerFot(**qwargs)
        else:
            raise ValueError("kjortetoy må være bil, sykkel eller fot.")

        self.directed = directed
        self._kostnad = kostnad
        self.search_tolerance = search_tolerance if search_tolerance else 100000000
        self.dist_faktor = dist_faktor
        self.kost_til_nodene = kost_til_nodene if kost_til_nodene else 0
        self._fjern_isolerte = fjern_isolerte

        # hent og klargjør nettverket for året og kommunene som er angitt
        if nettverk is None:
            self.nettverk = self.hent_nettverk()
        else:
            if not "source" in nettverk.columns or not "target" in nettverk.columns:
                raise ValueError(
                    "Finner ikke kolonnene 'source' og/eller 'target'. Kjør nettverket gjennom lag_nettverk() før Graf()"
                )
            self.nettverk = nettverk

        self.nettverk, self._noder = lag_node_ids(self.nettverk)

        if self._kjoretoy != "bil":
            if "stigningsprosent" in self.nettverk.columns:
                self.nettverk["minutter"] = self.beregn_minutter_stigning()
            else:
                self.nettverk["minutter"] = self.nettverk.length / (
                    self.fart * 16.666667
                )

        self._kostnad = self.sjekk_kostnad(kostnad)

        if not "isolert" in self.nettverk.columns:
            self._fjern_isolerte = False

        if self._kjoretoy == "bil":
            if not "sperring" in self.nettverk.columns:
                self.sperring = None

            if not "turn_restriction" in self.nettverk.columns:
                self.turn_restrictions = False

    def od_cost_matrix(
        self,
        startpunkter: gpd.GeoDataFrame,
        sluttpunkter: gpd.GeoDataFrame,
        id_kolonne=None,
        linjer=False,  # om man vil at rette linjer mellom start- og sluttpunktene returnert
        radvis=False,  # hvis False beregnes kostnaden fra alle startpunkter til alle sluttpunkter.
        cutoff: int = None,
        destination_count: int = None,
    ):
        self.nettverk = self.filtrer_nettverk()
        self.nettverk, self._noder = lag_node_ids(self.nettverk)

        return od_cost_matrix(
            self,
            startpunkter=startpunkter,
            sluttpunkter=sluttpunkter,
            id_kolonne=id_kolonne,
            linjer=linjer,
            radvis=radvis,
            cutoff=cutoff,
            destination_count=destination_count,
        )

    def service_area(
        self,
        startpunkter: gpd.GeoDataFrame,
        impedance,
        id_kolonne=None,
        dissolve=True,
    ):
        self.nettverk = self.filtrer_nettverk()
        self.nettverk, self._noder = lag_node_ids(self.nettverk)

        if not isinstance(self.kostnad, str):
            raise ValueError("Kan bare ha én kostnad (str) i shortest_path")

        return service_area(
            self,
            startpunkter=startpunkter,
            impedance=impedance,
            id_kolonne=id_kolonne,
            dissolve=dissolve,
        )

    def shortest_path(
        self,
        startpunkter: gpd.GeoDataFrame,
        sluttpunkter: gpd.GeoDataFrame,
        id_kolonne=None,
        cutoff: int = None,
        destination_count: int = None,
        radvis=False,
        tell_opp=False,
    ):
        self.nettverk = self.filtrer_nettverk()
        self.nettverk, self._noder = lag_node_ids(self.nettverk)

        if not isinstance(self.kostnad, str):
            raise ValueError("Kan bare ha én kostnad (str) i shortest_path")

        return shortest_path(
            self,
            startpunkter=startpunkter,
            sluttpunkter=sluttpunkter,
            id_kolonne=id_kolonne,
            cutoff=cutoff,
            destination_count=destination_count,
            radvis=radvis,
            tell_opp=tell_opp,
        )

    def kutt_linjer(self, meter) -> gpd.GeoDataFrame:
        self.nettverk = kutt_linjer(self.nettverk, meter)
        return self

    def hent_nettverk(self) -> gpd.GeoDataFrame:
        if int(self.aar) > NYESTE_AAR or int(self.aar) < ELDSTE_AAR:
            raise ValueError(f"aar må være mellom {ELDSTE_AAR} og {NYESTE_AAR}")

        if self.kjoretoy != "bil" and self.aar != NYESTE_AAR:
            raise ValueError(f"aar må være {NYESTE_AAR} for sykkel/fot")

        self.nettverkssti = self.nettverkssti.replace(str(NYESTE_AAR), str(self.aar))

        if self._kommuner:
            try:
                nettverk = les_geopandas(
                    self.nettverkssti, filters=[("KOMMUNENR", "in", self._kommuner)]
                )
            except Exception:
                kommuner = les_geopandas(KOMMUNESTI[0], layer=KOMMUNESTI[1]).rename(
                    columns={"kommunenummer": "KOMMUNENR"}
                )
                kommnr = (
                    [self._kommuner]
                    if isinstance(self._kommuner, str)
                    else self._kommuner
                )
                kommuner = kommuner.loc[
                    kommuner.KOMMUNENR.isin(kommnr), ["KOMMUNENR", "geometry"]
                ]

                nettverk = les_geopandas(self.nettverkssti)
                nettverk = nettverk.sjoin(kommuner, how="inner").drop(
                    "index_right", axis=1, errors="ignore"
                )

        else:
            nettverk = les_geopandas(self.nettverkssti)

        return nettverk.to_crs(25833)

    def filtrer_nettverk(self) -> gpd.GeoDataFrame:
        if self.kjoretoy == "bil":
            if self.directed and self.turn_restrictions:
                self.nettverk = self.nettverk[
                    self.nettverk["turn_restriction"] != False
                ]
            else:
                if "turn_restriction" in self.nettverk.columns:
                    self.nettverk = self.nettverk[
                        self.nettverk["turn_restriction"] != True
                    ]

            if self.sperring:
                self.nettverk = self.sett_opp_sperringer()

        if self.kjoretoy != "bil" and "typeveg" in self.nettverk.columns:
            self.nettverk = self.nettverk[
                ~self.nettverk.typeveg.isin(self.forbudte_vegtyper)
            ]

        if self.fjern_isolerte:
            self.nettverk = self.nettverk[self.nettverk["isolert"].fillna(0) == 0]

        if not len(self.nettverk):
            raise ValueError("Nettverket har 0 rader")

        return self.nettverk

    def beregn_minutter_stigning(self):
        minutter = self.nettverk.length / (self.fart * 16.666667)

        gange_med = (
            1
            + (
                self.prosent_straff_stigning
                * self.nettverk["stigningsprosent"].fillna(0)
            )
            / 100
        )
        dele_paa = (
            1
            + (
                self.prosent_straff_stigning
                * np.log(1 + abs(self.nettverk["stigningsprosent"].fillna(0)))
            )
            / 100
        )

        minutter_stigningsjustert = np.where(
            self.nettverk["stigningsprosent"].fillna(0) > 0,
            minutter * gange_med,
            minutter / dele_paa,
        )

        return minutter_stigningsjustert

    def sett_opp_sperringer(self) -> gpd.GeoDataFrame:
        if self.sperring is True or not "category" in self.nettverk.columns:
            return self.nettverk[self.nettverk["sperring"].fillna(0).astype(int) != 1]

        for vegkat in [vegkat.lower() for vegkat in self.sperring]:
            self.nettverk = self.nettverk[
                ~(
                    (self.nettverk["sperring"].fillna(0).astype(int) == 1)
                    & (self.nettverk["category"].str.lower() == vegkat)
                )
            ]

        return self.nettverk

    def sjekk_kostnad(self, kostnad):
        """sjekk om kostnadskolonnen finnes i dataene."""

        if isinstance(kostnad, str):
            kostnad = [kostnad]
        if not isinstance(kostnad, (list, tuple)):
            raise ValueError("kostnad må være string, liste eller tuple")

        kostnader = []
        for kost in kostnad:
            for kolonne in self.nettverk.columns:
                if kost in kolonne or kolonne in kost:
                    kostnader.append(kolonne)
                elif "min" in kost and "min" in kolonne:
                    kostnader.append(kolonne)
                elif "meter" in kost or "dist" in kost:
                    self.nettverk["meter"] = self.nettverk.length
                    kostnader.append("meter")

        if len(kostnader) == 0:
            if self.kjoretoy == "bil":
                raise ValueError("Finner ikke kostnadskolonne")
            else:
                kostnader = ["minutter"]

        kostnader = list(set(kostnader))

        if len(kostnader) > len(kostnad):
            raise ValueError(
                f"Flere enn {len(kostnad)} kolonner kan inneholde kostnaden{'e' if len(kostnad)>1 else ''} {', '.join(kostnad)}"
            )

        if len(kostnader) == 1:
            kostnader = kostnader[0]

        return kostnader

    def info(self) -> None:
        print("aar: ")
        print("nettverk: ")
        print("kostnad: ")
        print("kjoretoy: ")
        print("directed: ")
        print("turn_restrictions: ")
        print("search_tolerance: ")
        print("dist_faktor: ")
        print("kost_til_nodene: km/t ")

    # sørg for at kostnaden er brukbar
    @property
    def kostnad(self):
        return self._kostnad

    @kostnad.setter
    def kostnad(self, ny_kostnad):
        self._kostnad = self.sjekk_kostnad(ny_kostnad)
        return self._kostnad

    # hindre at fjern_isolerte settes til True når ikke isolert-kolonnen finnes
    @property
    def fjern_isolerte(self):
        return self._fjern_isolerte

    @fjern_isolerte.setter
    def fjern_isolerte(self, ny_verdi):
        if ny_verdi and not "isolert" in self.nettverk.columns:
            raise ValueError(
                "Kan ikke sette fjern_isolerte til True når kolonnen 'isolert' ikke finnes."
            )
        self._fjern_isolerte = ny_verdi
        return self._fjern_isolerte

    # disse skal det ikke være lov å endre
    @property
    def aar(self):
        return self._aar

    @property
    def kjoretoy(self):
        return self._kjoretoy

    @property
    def kommuner(self):
        return self._kommuner

    @property
    def noder(self):
        return self._noder

    def copy(self):
        return copy(self)

    def deepcopy(self):
        return deepcopy(self)

    # for å printe attributtene
    def __repr__(self) -> None:
        attrs = []
        for attr, val in self.__dict__.items():
            if attr in attrs:
                continue
            elif attr == "nettverkssti" or attr == "_noder":
                continue
            elif isinstance(val, (ReglerSykkel, ReglerFot, ReglerBil)):
                for attr, val in val.__dict__.items():
                    if attr == "nettverkssti":
                        pass
                    elif isinstance(val, str):
                        print(f"{attr.strip('_')} = '{val}',")
                    else:
                        print(f"{attr.strip('_')} = {val},")
                    attrs.append(attr)
                continue
            elif attr == "nettverk" and self.nettverkssti:
                print(f"{attr} = GeoDataFrame hentet fra: {self.nettverkssti},")
            elif isinstance(val, gpd.GeoDataFrame):
                print(f"{attr.strip('_')} = GeoDataFrame,")
            elif isinstance(val, str):
                print(f"{attr.strip('_')} = '{val}',")
            else:
                print(f"{attr.strip('_')} = {val},")
            attrs.append(attr)
        del attrs
        return ""

    # for å gjøre regel-atributtene tilgjengelig direkte i Graf-objektet.
    def __getattr__(self, navn):
        return self.regler.__getattribute__(navn)
