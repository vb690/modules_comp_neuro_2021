__docformat__ = "google"

import os

import pandas as pd

import numpy as np

from tqdm import tqdm


class ExperimentParser:
    """A Class parsing and holding experiment results.

    Attributes:
        experiment_dir (str): string specifying where the .dat files to be
            parsed are located.
    """
    def __init__(self, experiment_dir):
        """Instantiate ExperimentParser with experiment_dir
        """
        self.experiment_dir = experiment_dir

        # this are created empty when the parser is instatiated
        self.subject_code = None
        self.counterbalance_code = None
        self.task_format_code = None
        self.condition_code = None
        self.phase_code = None

    @staticmethod
    def get_logs(line, is_test):
        """Static method for parsing and standardizing experiment logs from
        the .dat file.

            Args:
                line (list): list containing the the experiment log for a
                    a trial in the .dat file.
                is_test (bool): bolean specifying if the log is coming from the
                    training or test phase.

            Returns:
                logs (list): a list of floats containing the parsed experiment
                    log.
        """
        # test mode has a different format
        # and doesn't have outcome or reward
        if is_test:
            line.insert(5, None)
            line.insert(7, None)
        else:
            # in training mode
            # we have to infer response given
            # from response rewarded
            line.insert(6, None)
        # if at this point len(line) < 9
        # response time is missing (AnoF condition)
        while len(line) < 9:
            line.append(None)
        logs = [np.float64(log) if log is not None else log for log in line]
        return logs

    @staticmethod
    def get_prob_norm_correct(cues,
                              mapping={0: 0.25, 1: 0.25, 2: -0.25, 3: -0.25}):
        """Compute the probability of a normative correct response in a trial
        given cues probability mapping.

            Args:
                cues (array): a binary numpy array reporting which cues were
                    present in the trial.
                mapping (dict): a dictionary with keys indicating the index of
                    a specific cue in cues (e.g. 0 is cue in position 0) and
                    values the changes in probability for a specific outcome to
                    occour.

            Returns:
                p (float): a float specifying the probability of a target
                    outcome to happen.
        """
        mapping = np.array([mapping[cue] for cue in range(len(mapping))])
        cues = np.argwhere(cues.values == 1).flatten()
        p = 0.5 + mapping[cues].sum()
        return p

    @staticmethod
    def get_norm_correct(prob_norm_correct):
        """Get the normatively correct response given a probability.

            Args:
                prob_norm_correct (float): a float specifying the probability
                    of a target outcome to happen.

            Returns:
                norm_correct (int): integer or None specifying which of two
                    target outcomes is the normatively correct one.
        """
        if prob_norm_correct < 0.5:
            return 2
        elif prob_norm_correct > 0.5:
            return 1
        else:
            return None

    def get_subject_data(self, text):
        """Turn text data extracted from a subject
        .dat file in a format suitable to be transformed in a pandas DataFrame.

            Args:
                text (list): is a list of strings containg the logs from all
                    the trials extracted from the .dat file.

            Returns:
                subject_data (list): list of lists containing all the trails
                    logs extracted for a specific subject.
        """
        subject_data = []
        fnoa_order = None

        for line in tqdm(text, leave=False):

            if '%' in line:  # ignore values with % as in forum
                continue
            elif 'Map' in line:
                continue
            # APA recommend the use of participant tho
            elif 'Subject' in line:
                self.subject_code = line.split()[0]
            elif 'Counterbal' in line:
                self.counterbalance_code = line.split()[0]
            elif 'Format' in line:
                self.task_format_code = line.split()[0]
            elif 'Condition' in line:
                self.condition_code = line.split()[0]
                # we could verify which counterbal condition
                # has FnoA first but it would be kinda cheating
                if fnoa_order is None:
                    fnoa_order = 1 if \
                        self.condition_code == 'FnoA' else 2
            elif 'Training' in line:
                self.phase_code = line.split()[0]
            elif 'Test' in line:
                self.phase_code = line.split()[0]
            else:
                experiment_info = [
                    self.subject_code,
                    self.counterbalance_code,
                    self.task_format_code,
                    self.condition_code,
                    self.phase_code,
                    fnoa_order
                ]
                experiment_logs = self.get_logs(
                    line=line.split(),
                    is_test=self.phase_code == 'Test'
                )
                subject_data.append(experiment_info + experiment_logs)

        return subject_data

    def parse_dat_files(self):
        """Parse the .dat file into a Pandas DataFrame and setting it as
        an attributed to the parser.

            Args:
                None

            Returns:
                None
        """
        experiment_data = []

        for filename in tqdm(os.listdir(self.experiment_dir)):

            with open(f'{self.experiment_dir}{filename}') as infile:

                text = infile.read()
                text = text.split('\n')
                text = [
                    txt_str.strip() for txt_str in text if txt_str != ''
                ]

                subject_data = self.get_subject_data(text=text)

                subject_data = pd.DataFrame(
                    np.array(subject_data),
                    columns=[
                        'subno',
                        'counterbal',
                        'format',
                        'condition',
                        'phase',
                        'fnoa_order',
                        'trial_n',
                        'cue_1',
                        'cue_2',
                        'cue_3',
                        'cue_4',
                        'outcome',
                        'resp_given',
                        'resp_rewar',
                        'resp_time'
                    ]
                )
                # we standardize response encoding because it is infuriating
                # so now 1 is 1 in every field and 2 is 2 in every field
                subject_data['outcome'] = subject_data['outcome'].map(
                    {0: 2, 1: 1}
                )

            experiment_data.append(subject_data)

        experiment_data = pd.concat(
            experiment_data,
            ignore_index=True
        )

        # compute the normatively correct response
        experiment_data['prob_outcome_1'] = experiment_data[
            [f'cue_{cue_number}' for cue_number in range(1, 5)]
        ].apply(
            self.get_prob_norm_correct,
            axis=1
        )
        experiment_data['resp_norm_corr'] = \
            experiment_data['prob_outcome_1'].apply(
                self.get_norm_correct
            )

        # define the include vector
        # hard-code this since it is tricky to programmatically alter
        # a slice of pandas DataFrame over multiple conditions
        experiment_data['include'] = 1
        experiment_data.loc[
            (experiment_data['condition'] == 'FnoA') &
            (experiment_data['trial_n'].isin([53, 54, 55, 56])) &
            (experiment_data['fnoa_order'] == 1),
            'include'
        ] = 0
        experiment_data.loc[
            experiment_data['include'] == 0, 'resp_rewar'
        ] = None

        # infer the response given based on reward
        experiment_data.loc[
            experiment_data['resp_rewar'] == 1, 'resp_given'
        ] = experiment_data['outcome']
        # when it is not rewarded we
        # invert the response given
        experiment_data.loc[
            experiment_data['resp_rewar'] == 0, 'resp_given'
        ] = experiment_data['outcome'].map(
            {
                1: 2, 2: 1
            }
        )

        # re arrange columns
        experiment_data = experiment_data[
            [
                'subno',
                'fnoa_order',
                'format',
                'counterbal',
                'condition',
                'trial_n',
                'phase',
                'include',
                'cue_1',
                'cue_2',
                'cue_3',
                'cue_4',
                'prob_outcome_1',
                'outcome',
                'resp_given',
                'resp_norm_corr',
                'resp_time',
                'resp_rewar'
            ]
        ]

        experiment_data = experiment_data.sort_values(
            ['subno', 'condition', 'trial_n']
        ).reset_index(drop=True)
        setattr(self, 'experiment_data', experiment_data)

        return None

    def get_data(self, filters=None):
        """Get the stored parsed experiment data applying a series of filters
        if sepcified. A copy of the data is always returned.

            Args:
                filters (dict): a dictionary with keys indicating the column
                    on which the filter is applied and values specifying the
                    value thatthe column has to match. The filters are applied
                    in sequence but order cannot be guaranteed.

            Returns:
                filtered_data (DataFrame): a pandas DataFrame with the
                    experiment data.
        """
        filtered_data = self.experiment_data.copy()
        if filters is not None:
            for column_name, value in filters.items():

                filtered_data = filtered_data[
                    filtered_data[column_name] == value
                ]

            return filtered_data
        else:
            return filtered_data


def merge_sav_file(left_df, path, keys):
    """Utility function for merging a pandas DataFrame with sav file
    on a list of keys (i.e. columns).

        Args:
            left_df (DataFrame): a pandas DataFrame that wil be joined with
                the .sav file.
            path (str): a string specifying the location of the .sav file.
            keys (list): a list of strings specifying on which columns the
                .sav file will be joined.

        Returns:
            merged (DataFrame): a pandas DataFrame resulting from joining
                left_df with the sav file on keys.
    """
    sav_df = pd.read_spss(
        path
    )
    merged = pd.merge(
        left_df,
        sav_df,
        how='left',
        on=keys
    )
    return merged
