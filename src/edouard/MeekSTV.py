#from ..votekit.elections.election_types.ranking.abstract_ranking import RankingElection
#from ..votekit.elections.transfers import fractional_transfer
#from ..votekit.pref_profile import RankProfile, ProfileError
#from ..votekit.elections.election_state import ElectionState
#from ..votekit.ballot import RankBallot
#from ..votekit.cleaning import (
#    remove_and_condense_rank_profile,
#    remove_cand_rank_ballot,
#    condense_rank_ballot,
#)
#from ..votekit.utils import (
#    _first_place_votes_from_df_no_ties,
#    ballots_by_first_cand,
#    tiebreak_set,
#    elect_cands_from_set_ranking,
#    score_dict_to_ranking,
#)

from .permutations import update_perm_idx_vectorized, WeightVectorCalculator

from typing import Optional, Callable, Union, List
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from itertools import groupby
from .utils import convert_pf_to_numpy_arrays


class MeekSTV:

    def __init__(
        self,
        profile,
        m: int = 1,
        tiebreak: Optional[str] = None,
    ):
        self.profile = profile
        self.m = m

        #self._ballot_length = profile.max_ranking_length
        #self._winner_tiebreak = tiebreak
        #self._loser_tiebreak = tiebreak if tiebreak is not None else "first_place"

        self.candidates = list(profile.candidates)

        self._ballot_matrix, self._mult_vec, self._fpv_vec = convert_pf_to_numpy_arrays(profile)

        self._core = MeekCore(
            ballot_matrix=self._ballot_matrix,
            mult_vec=self._mult_vec.copy(),
            num_cands=len(self.candidates),
            m=m)
        
        self._fpv_by_round, self._play_by_play, self._tiebreak_record = self._core._run_core()

class MeekCore:
    def __init__(
        self,
        ballot_matrix: NDArray,
        mult_vec: NDArray,
        num_cands: int,
        m: int = 1,
        pos_vec: NDArray = None,
        fpv_vec: NDArray = None,
        winner_combination_vec: NDArray = None,
        bool_ballot_matrix: NDArray = None,
        winner_to_cand: List = [],
        initial_losers: List = [],
        tiebreak: Optional[str] = None,
        tolerance: float = 1e-6,
        epsilon: float = 1e-6,
        max_iterations: int = 500,
    ):
        """
        The play-by-play logs some information for the public methods:
                    - round number
                    - list of candidates elected or eliminated this round
                    - keep factors of all winners in this round
                    - quota this round (after calibration)
                    - number of iterations used to calibrate keep factors & quota
                    - round type: 'election' or 'elimination'
        """
        self._ballot_matrix = ballot_matrix
        self._mult_vec = mult_vec
        self._num_cands = num_cands
        self.m = m
        self.tolerance = tolerance
        self.epsilon = epsilon
        self._max_iterations = max_iterations

        self._winner_tiebreak = tiebreak
        self._loser_tiebreak = tiebreak if tiebreak is not None else "first_place"

        self._initial_pos_vec, self._initial_fpv_vec, self._initial_winner_comb_vec, self._initial_bool_ballot_matrix = self.update_helpers(
            pos_vec,
            fpv_vec,
            winner_combination_vec,
            bool_ballot_matrix,
            winner_to_cand,
            new_losers=initial_losers,
        )
        self._initial_winner_to_cand = winner_to_cand

        self._wt_calculator = WeightVectorCalculator(self.m-1, self.m)

        self._fpv_by_round, self._play_by_play, self._tiebreak_record = self._run_core()
        #self.election_states = self._make_election_states()

    def update_helpers(self,
                   pos_vec = None,
                   fpv_vec = None,
                   winner_comb_vec = None,
                   bool_ballot_matrix = None,
                   winner_to_cand = [],
                   new_losers = []):
        num_ballots = self._ballot_matrix.shape[0]
        if pos_vec is None:
            pos_vec = np.zeros(num_ballots, dtype=np.int8)
        #assert len(pos_vec) == num_ballots 
        if fpv_vec is None:
            fpv_vec = self._ballot_matrix[np.arange(self._ballot_matrix.shape[0]), pos_vec]
        if winner_comb_vec is None:
            winner_comb_vec = np.zeros(num_ballots)
        if bool_ballot_matrix is None:
            bool_ballot_matrix = self._ballot_matrix != -127
        if len(new_losers)>0:
            bool_ballot_matrix &= ~np.isin(self._ballot_matrix, new_losers)
            needs_update = np.isin(fpv_vec, new_losers)
            pos_vec[needs_update] = np.argmax(bool_ballot_matrix[needs_update, :], axis = 1)
            fpv_vec[needs_update] = self._ballot_matrix[needs_update, pos_vec[needs_update]]
        for _ in range(len(winner_to_cand)):
            needs_update = np.isin(fpv_vec, winner_to_cand)
            winner_comb_vec[needs_update] = update_perm_idx_vectorized(winner_comb_vec[needs_update], fpv_vec[needs_update], self.m)
            bool_ballot_matrix[needs_update, pos_vec[needs_update]] = False
            pos_vec[needs_update] = np.argmax(bool_ballot_matrix[needs_update,:], axis = 1)
            fpv_vec[needs_update] = self._ballot_matrix[needs_update, pos_vec[needs_update]]
        return pos_vec, fpv_vec, winner_comb_vec, bool_ballot_matrix
    
    def tally_calculator_engine(self, fpv_vec, winner_combination_vec, keep_factors, winner_to_cand):
        unique_winner_combos = np.unique(winner_combination_vec)
        overall_tallies = np.zeros(self._num_cands)
        for perm_idx in unique_winner_combos:
            ballot_mask = winner_combination_vec == perm_idx
            non_exhausted_mask = ballot_mask & (fpv_vec >=0)

            wt_vec_for_this_permutation = self._wt_calculator.make_wt_vec(perm_idx, keep_factors)
            weights_per_winner=sum(self._mult_vec[ballot_mask])*wt_vec_for_this_permutation[:-1]

            leftover_weight = wt_vec_for_this_permutation[-1]
            leftover_tally =np.bincount(fpv_vec[non_exhausted_mask], weights=self._mult_vec[non_exhausted_mask]*leftover_weight, minlength=self._num_cands)
            leftover_tally[winner_to_cand] += weights_per_winner[:len(winner_to_cand)]
            overall_tallies += leftover_tally
        return overall_tallies

    def calibrate_keep_factors(self, fpv_vec, winner_combination_vec, winner_to_cand):
        L = len(winner_to_cand)
        keep_factors = np.ones(L)
        for iteration in range(self._max_iterations):
            tallies = self.tally_calculator_engine(fpv_vec, winner_combination_vec, keep_factors, winner_to_cand)
            quota = sum(tallies) / (self.m+1) + self.epsilon
            if np.any(tallies[winner_to_cand] < quota):
                raise ValueError(f"Tally for a winning candidate is below quota: {tallies[winner_to_cand]} < {quota}, keep factors: {keep_factors}")
            if np.all(tallies[winner_to_cand] - quota < self.tolerance):
                break
            new_keep_factors = quota/tallies[winner_to_cand]
            keep_factors *= new_keep_factors
        return tallies, keep_factors, iteration+1, quota
    
    def meek_stv_engine(self, 
                        pos_vec,
                        fpv_vec, 
                        winner_combination_vec,
                        bool_ballot_matrix,
                        winner_to_cand,
                        hopeful):
        #1) calibrate keep factors
        #2) record info and determine loser/winner(s)
        #3) update helpers
        tallies, keep_factors, iterations, current_quota = self.calibrate_keep_factors(
            fpv_vec, 
            winner_combination_vec,
            winner_to_cand
        )
        non_winner_tallies = np.copy(tallies)
        non_winner_tallies[winner_to_cand] = -1
        if np.any(non_winner_tallies > current_quota):
            new_winners = np.where(non_winner_tallies > current_quota)[0].tolist()
            print("New winners:", new_winners)
            winner_to_cand.extend(new_winners)
            round_type = "winner"
            new_losers = []
        else:
            hopeful_tallies = tallies[hopeful]
            new_loser = int(hopeful[int(np.argmin(hopeful_tallies))])
            hopeful.remove(new_loser)
            round_type = "loser"
            new_losers = [new_loser]
            print("New loser:", new_loser)
            new_winners = []
        if len(winner_to_cand) < self.m:
            pos_vec, fpv_vec, winner_combination_vec, bool_ballot_matrix = self.update_helpers(
                pos_vec=pos_vec,
                fpv_vec=fpv_vec,
                winner_comb_vec=winner_combination_vec,
                bool_ballot_matrix=bool_ballot_matrix,
                winner_to_cand=winner_to_cand,
                new_losers=new_losers
            )
        return (tallies, keep_factors, iterations, current_quota, 
                pos_vec, fpv_vec, winner_combination_vec, 
                bool_ballot_matrix, round_type, new_losers, new_winners, winner_to_cand, hopeful)
    
    def _run_core(self):
        fpv_by_round = []
        play_by_play = []
        tiebreak_record = []

        pos_vec = self._initial_pos_vec.copy()
        fpv_vec = self._initial_fpv_vec.copy()
        winner_combination_vec = self._initial_winner_comb_vec.copy()
        bool_ballot_matrix = self._initial_bool_ballot_matrix.copy()
        winner_to_cand = self._initial_winner_to_cand.copy()

        hopeful = np.arange(0, self._num_cands).tolist()

        round_number = 0
        while len(winner_to_cand) < self.m:
            (
                tallies,
                keep_factors,
                iterations,
                current_quota,
                pos_vec,
                fpv_vec,
                winner_combination_vec,
                bool_ballot_matrix,
                round_type,
                new_losers,
                new_winners,
                winner_to_cand,
                hopeful,
            ) = self.meek_stv_engine(
                pos_vec,
                fpv_vec,
                winner_combination_vec,
                bool_ballot_matrix,
                winner_to_cand,
                hopeful,
            )
            fpv_by_round.append(tallies.copy())
            winners_or_losers = new_losers if round_type == "loser" else new_winners
            #tiebreak_record.append #TODO: tiebreaks
            play_by_play.append(
                {
                    "round_number": round_number,
                    "new_winners_or_losers": winners_or_losers,
                    "keep_factors": keep_factors.copy(),
                    "quota": current_quota,
                    "iterations": iterations,
                    "round_type": round_type,
                }
            )
            round_number += 1

        return fpv_by_round, play_by_play, tiebreak_record
    
#buncha profiles:

#deg2_profile = PreferenceProfile(
#        ballots=(
#            Ballot(ranking=(frozenset({"A"}), frozenset({"B"}), frozenset({"C"})), weight=90),
#            Ballot(ranking=(frozenset({"B"}), frozenset({"A"}), frozenset({"C"})), weight=90),
#            Ballot(ranking=(frozenset({"A"}), frozenset({"C"})), weight=30),
#            Ballot(ranking=(frozenset({"B"}), frozenset({"C"})), weight=30),
#            Ballot(ranking=(frozenset({"C"}),), weight=61),
#            Ballot(ranking=(frozenset({"D"}),), weight=99),
#        ),
#        candidates=("A", "B", "C", "D"),
#    )
#
#many_exhaust_profile = PreferenceProfile(
#        ballots=(
#            Ballot(ranking=(frozenset({"A"}), frozenset({"C"})), weight=120),
#            Ballot(ranking=(frozenset({"B"}),), weight=102),
#            Ballot(ranking=(frozenset({"C"}),), weight=77),
#            Ballot(ranking=(frozenset({"D"}),), weight=60),
#            Ballot(ranking=(frozenset({"E"}),), weight=51),
#            Ballot(ranking=(frozenset({"F"}),), weight=40),
#        ),
#        candidates=("A", "B", "C", "D", "E", "F"),
#    )
#
#squeeze_profile = PreferenceProfile(
#        ballots=(
#            Ballot(ranking=(frozenset({"A"}), frozenset({"B"})), weight=60),
#            Ballot(ranking=(frozenset({"A"}),), weight=90),
#            Ballot(ranking=(frozenset({"B"}),), weight=63),
#            Ballot(ranking=(frozenset({"C"}),), weight=87),
#        ),
#        candidates=("A", "B", "C"),
#    )
#
#fractional_not_same_as_meek_profile = PreferenceProfile(
#        ballots=(
#            Ballot(ranking=(frozenset({"A"}), frozenset({"B"})), weight=70),
#            Ballot(ranking=(frozenset({"B"}),), weight=80),
#            Ballot(ranking=(frozenset({"C"}),), weight=85),
#            Ballot(ranking=(frozenset({"D"}), frozenset({"A"})), weight=35),
#            Ballot(ranking=(frozenset({"E"}), frozenset({"A"})), weight=30),
#        ),
#        candidates=("A", "B", "C", "D", "E"),
#    )
#
#election = MeekCore(ballot_matrix = ballot_matrix,
#                        mult_vec = mult_vec,
#                        num_cands = 6,
#                        m = 2,
#                        max_iterations = 50,)
#
#print(election._fpv_by_round)
#print(election._play_by_play)