# partly adapted from D3Rec: https://github.com/c0natus/D3Rec
import os
import json
import random
from datetime import datetime
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

## usage 
class FairnessDataGenerator:
    
    def __init__(self, dataset_name, n_users, n_items, train_data, data_root, short_head_ratio=0.2, original_user_ids=None):
        self.dataset_name = dataset_name.lower()
        self.n_users = n_users
        self.n_items = n_items
        self.train_data = train_data
        self.data_root = data_root
        self.short_head_ratio = short_head_ratio
        self.original_user_ids = original_user_ids
        self.item_groups = None
        self.user_groups = None
        self.stats = {}
        self.user_hist_matrix = None
        self.user_histories = None
    
    def interaction_list_to_matrix(self):
        user_ids = self.train_data[:, 0].astype(int)
        item_ids = self.train_data[:, 1].astype(int)
        data = np.ones(len(user_ids))
        matrix = sp.csr_matrix((data, (user_ids, item_ids)), shape=(self.n_users, self.n_items))
        return matrix
    
    def _gini_from_counts(self, counts: np.ndarray) -> float:
        counts = np.asarray(counts, dtype=np.float64).copy()
        if counts.ndim != 1:
            counts = counts.reshape(-1)

        # guard: empty or all zeros
        n = counts.size
        if n == 0:
            return 0.0
        total = counts.sum()
        if total <= 0:
            return 0.0

        # sort ascending (required)
        counts.sort()

        i = np.arange(1, n + 1, dtype=np.float64)  # 1..n
        weights = (n + 1 - i) / (n + 1)            # (n ... 1)/(n+1)
        gini = 1.0 - 2.0 * np.sum(weights * (counts / total))

        if gini < 0:
            gini = 0.0
        if gini > 1:
            gini = 1.0
        return float(gini)

    def compute_gini_iu(self):
        train_matrix = self.interaction_list_to_matrix().tocsr()

        item_counts = np.array(train_matrix.sum(axis=0)).flatten()
        user_counts = np.array(train_matrix.sum(axis=1)).flatten()

        gini_i = self._gini_from_counts(item_counts)
        gini_u = self._gini_from_counts(user_counts)

        self.stats['gini'] = {
            'gini_i': float(gini_i),
            'gini_u': float(gini_u),
            'note': 'Computed from train URM interaction counts (item_counts and user_counts).'
        }

        print("\n" + "-"*80)
        print("Gini Statistics (Train URM)")
        print("-"*80)
        print(f"  Gini_i (item interaction inequality): {gini_i:.6f}")
        print(f"  Gini_u (user interaction inequality): {gini_u:.6f}")

        return {'gini_i': gini_i, 'gini_u': gini_u}

    def compute_gini_full(self, valid_data=None, test_data=None):
        # Combine all splits
        parts = [self.train_data]
        if valid_data is not None:
            parts.append(valid_data)
        if test_data is not None:
            parts.append(test_data)
        full_data = np.concatenate(parts, axis=0)

        user_ids = full_data[:, 0].astype(int)
        item_ids = full_data[:, 1].astype(int)
        data = np.ones(len(user_ids))
        full_matrix = sp.csr_matrix((data, (user_ids, item_ids)),
                                    shape=(self.n_users, self.n_items))

        item_counts = np.array(full_matrix.sum(axis=0)).flatten()
        user_counts = np.array(full_matrix.sum(axis=1)).flatten()

        gini_i = self._gini_from_counts(item_counts)
        gini_u = self._gini_from_counts(user_counts)

        self.stats['gini_full'] = {
            'gini_i': float(gini_i),
            'gini_u': float(gini_u),
            'num_interactions': int(len(full_data)),
            'note': 'Computed from full dataset (train+valid+test) interaction counts.'
        }

        print("\n" + "-"*80)
        print("Gini Statistics (Full Dataset: train+valid+test)")
        print("-"*80)
        print(f"  Total interactions: {len(full_data)}")
        print(f"  Gini_i (item interaction inequality): {gini_i:.6f}")
        print(f"  Gini_u (user interaction inequality): {gini_u:.6f}")

        return {'gini_i_full': gini_i, 'gini_u_full': gini_u}

    def _merge_ftky_profiles(self, nyc_path, tky_path, output_path):
        print(f"\n  Merging FTKY user profiles...")
        
        nyc_df = pd.read_csv(nyc_path, sep=' ', header=None, 
                            names=['user_id', 'gender', 'twitter_friend_count', 'twitter_follower_count'])
        print(f"    NYC users: {len(nyc_df)}")
        
        tky_df = pd.read_csv(tky_path, sep=' ', header=None,
                            names=['user_id', 'gender', 'twitter_friend_count', 'twitter_follower_count'])
        print(f"    Tokyo users: {len(tky_df)}")
        
        merged_df = pd.concat([nyc_df, tky_df], ignore_index=True)
        merged_df = merged_df.drop_duplicates(subset=['user_id'], keep='first')
        
        print(f"    Merged users: {len(merged_df)} (after deduplication)")
        
        merged_df.to_csv(output_path, sep=' ', header=False, index=False)
        print(f"    Saved merged profile to: {output_path}")
        
        return output_path

    def create_item_groups_from_popularity(self):
        print("\n" + "-"*80)
        print("Creating Item Groups (Popularity-based)")
        print("-"*80)
        
        # Convert to matrix format
        train_matrix = self.interaction_list_to_matrix()
        
        # Calculate item popularity
        item_popularity = np.array(train_matrix.sum(axis=0)).flatten()
        
        # Sort items by popularity (descending)
        sorted_indices = np.argsort(item_popularity)[::-1]
        
        # Create groups: 1 = short-head, 2 = long-tail
        item_groups = np.ones(self.n_items, dtype=int) * 2  # Default to long-tail
        n_short_head = int(self.n_items * self.short_head_ratio)
        item_groups[sorted_indices[:n_short_head]] = 1  # Top items are short-head
        
        # Statistics
        short_head_pop = item_popularity[item_groups == 1]
        long_tail_pop = item_popularity[item_groups == 2]
        
        print(f"  Total items: {self.n_items}")
        print(f"  Short-head items (group 1): {(item_groups == 1).sum()} ({self.short_head_ratio*100:.1f}%)")
        print(f"  Long-tail items (group 2): {(item_groups == 2).sum()} ({(1-self.short_head_ratio)*100:.1f}%)")
        print(f"\n  Popularity Statistics:")
        print(f"    Short-head avg interactions: {short_head_pop.mean():.2f}")
        print(f"    Long-tail avg interactions: {long_tail_pop.mean():.2f}")
        print(f"    Short-head min interactions: {short_head_pop.min():.0f}")
        print(f"    Long-tail max interactions: {long_tail_pop.max():.0f}")
        
        self.item_groups = item_groups
        self.stats['item_groups'] = {
            'method': 'popularity',
            'short_head_ratio': self.short_head_ratio,
            'group_1_count': int((item_groups == 1).sum()),
            'group_1_label': 'Short-head',
            'group_2_count': int((item_groups == 2).sum()),
            'group_2_label': 'Long-tail',
            'short_head_avg_interactions': float(short_head_pop.mean()),
            'long_tail_avg_interactions': float(long_tail_pop.mean())
        }

        return item_groups
    
    def create_user_groups_from_gender_ml1m(self, users_file):
        """
        Create user groups based on gender from MovieLens-1M users.dat file
        Format: UserID::Gender::Age::Occupation::Zip-code
        """
        print(f"\n  Reading gender data from: {os.path.basename(users_file)}")
        
        user_groups = np.zeros(self.n_users, dtype=int)
        
        with open(users_file, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('::')
                if len(parts) >= 2:
                    user_id = int(parts[0])
                    gender = parts[1]
                    
                    user_idx = user_id - 1
                    
                    if user_idx < self.n_users:
                        user_groups[user_idx] = 1 if gender == 'M' else 2
        
        male_count = (user_groups == 1).sum()
        female_count = (user_groups == 2).sum()
        unknown_count = (user_groups == 0).sum()
        
        print(f"  Male users (group 1): {male_count} ({male_count/self.n_users*100:.1f}%)")
        print(f"  Female users (group 2): {female_count} ({female_count/self.n_users*100:.1f}%)")
        if unknown_count > 0:
            print(f"  WARNING: {unknown_count} users have missing gender information!")
        
        self.stats['user_groups'] = {
            'method': 'gender_ml1m',
            'source_file': os.path.basename(users_file),
            'group_1_count': int(male_count),
            'group_1_label': 'Male',
            'group_2_count': int(female_count),
            'group_2_label': 'Female',
            'missing_count': int(unknown_count)
        }
        
        return user_groups
    
    def create_user_groups_from_gender_ftky(self, profile_file):
        """
        Create user groups based on gender from FTKY user profile file
        Format: UserID Gender TwitterFriendCount TwitterFollowerCount
        Each field is separated by a space
        """
        print(f"\n  Reading gender data from: {os.path.basename(profile_file)}")
        if self.original_user_ids is None:
            raise ValueError("original_user_ids is required to map profile user_id to encoded indices. "
                             "Pass le_user.classes_ when constructing FairnessDataGenerator.")
        orig_to_enc = {int(orig): enc for enc, orig in enumerate(self.original_user_ids)}
        user_groups = np.zeros(self.n_users, dtype=int)
        
        matched_users = 0
        with open(profile_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        user_id = int(parts[0])
                        gender = parts[1].strip().upper()
                        if user_id in orig_to_enc:
                            idx = orig_to_enc[user_id]
                            if gender in ['M', 'MALE', '0']:
                                user_groups[idx] = 1
                                matched_users += 1
                            elif gender in ['F', 'FEMALE', '1']:
                                user_groups[idx] = 2
                                matched_users += 1
                    except ValueError:
                        continue
        
        male_count = (user_groups == 1).sum()
        female_count = (user_groups == 2).sum()
        unknown_count = (user_groups == 0).sum()
        
        print(f"  Matched users with gender info: {matched_users}")
        print(f"  Male users (group 1): {male_count} ({male_count/self.n_users*100:.1f}%)")
        print(f"  Female users (group 2): {female_count} ({female_count/self.n_users*100:.1f}%)")
        if unknown_count > 0:
            print(f"  WARNING: {unknown_count} users have missing gender information!")
        
        self.stats['user_groups'] = {
            'method': 'gender_ftky',
            'source_file': os.path.basename(profile_file),
            'matched_users': int(matched_users),
            'group_1_count': int(male_count),
            'group_1_label': 'Male',
            'group_2_count': int(female_count),
            'group_2_label': 'Female',
            'missing_count': int(unknown_count)
        }
        
        return user_groups
        
    
    def create_user_groups_from_activity(self, activity_threshold='median'):
        """
        Create user groups based on user activity level (fallback method)
        """
        print("\n  Creating user groups based on activity level")
        
        train_matrix = self.interaction_list_to_matrix()
        user_activity = np.array(train_matrix.sum(axis=1)).flatten()
        
        if activity_threshold == 'median':
            threshold = np.median(user_activity)
        elif activity_threshold == 'mean':
            threshold = np.mean(user_activity)
        else:
            threshold = float(activity_threshold)
        
        user_groups = np.where(user_activity >= threshold, 1, 2).astype(int)
        
        print(f"  Threshold: {threshold:.2f} interactions")
        print(f"  High-activity users (group 1): {(user_groups == 1).sum()} ({(user_groups == 1).sum()/self.n_users*100:.1f}%)")
        print(f"  Low-activity users (group 2): {(user_groups == 2).sum()} ({(user_groups == 2).sum()/self.n_users*100:.1f}%)")
        
        self.stats['user_groups'] = {
            'method': 'activity',
            'threshold': float(threshold),
            'group_1_count': int((user_groups == 1).sum()),
            'group_1_label': 'High-activity',
            'group_2_count': int((user_groups == 2).sum()),
            'group_2_label': 'Low-activity'
        }
        
        return user_groups

    def create_popularity_bins(self, short_head_ratio=0.2, tie_break='stable', round_to='up'):
        """
        Create three popularity bins by mass (interaction counts), not by item counts.

        Bin 1 (label=1): Short-head — take most popular items from the top until
                         cumulative interactions reach r * total_interactions.
        Bin 3 (label=3): tail  — take least popular items from the bottom until
                         its cumulative interactions >= Bin 1's cumulative interactions.
        Bin 2 (label=2): Middle     — all remaining items.

        Args:
            short_head_ratio (float or None): ratio r of total interaction mass for Bin1;
                                              if None, use self.short_head_ratio.
            tie_break (str): 'stable' (deterministic) or 'random' to randomize ties.
            round_to (str): 'up' or 'down' for the cutoff rounding when cumulative equals threshold.

        Returns:
            bins (np.ndarray): shape (n_items,), integer labels in {1,2,3}
            item_popularity (np.ndarray): shape (n_items,), interaction counts per item
            indices (dict): {'bin1': idx_head, 'bin2': idx_mid, 'bin3': idx_tail}
        """
        print("\n" + "-" * 80)
        print("Creating Popularity Bins by MASS (Bin1: Short-head mass r, Bin3: Tail mass ≈ Bin1)")
        print("-" * 80)

        r = float(short_head_ratio) if short_head_ratio is not None else float(self.short_head_ratio)
        r = max(0.0, min(0.9, r))

        train_matrix = self.interaction_list_to_matrix()
        item_popularity = np.array(train_matrix.sum(axis=0)).flatten()  # (n_items,)
        total_mass = item_popularity.sum()

        n = self.n_items
        bins = np.full(n, 2, dtype=int)  # default Bin2 (middle)

        if total_mass == 0:
            print("  WARNING: total interaction mass is 0; all items assigned to Bin2.")
            self.stats['popularity_bins_mass'] = {
                'short_head_ratio': r,
                'bin1_mass': 0.0, 'bin2_mass': 0.0, 'bin3_mass': 0.0,
                'bin1_count': 0, 'bin2_count': int(n), 'bin3_count': 0
            }
            return bins, item_popularity, {'bin1': np.array([], int), 'bin2': np.arange(n), 'bin3': np.array([], int)}

        if tie_break == 'random':
            rng = np.random.default_rng(42)
            perm = rng.permutation(n)
            sorted_desc = np.lexsort((perm, -item_popularity))
        else:
            sorted_desc = np.argsort(item_popularity, kind='stable')[::-1]

        target_mass_head = r * total_mass
        cum = 0
        cut = 0
        for k, idx in enumerate(sorted_desc):
            cum += item_popularity[idx]
            cut = k
            if cum >= target_mass_head:
                break
        head_end = cut if round_to == 'up' else max(0, cut - 1)
        head_idx = sorted_desc[:head_end + 1]
        head_mass = item_popularity[head_idx].sum()

        sorted_asc = sorted_desc[::-1]  # ascending (from least popular to most popular)
        cum_tail = 0
        cut_tail = 0
        for k, idx in enumerate(sorted_asc):
            cum_tail += item_popularity[idx]
            cut_tail = k
            if cum_tail >= head_mass:
                break
        tail_end = cut_tail
        tail_idx = sorted_asc[:tail_end + 1]
        tail_mass = item_popularity[tail_idx].sum()

        bins[head_idx] = 1
        bins[tail_idx] = 3

        mid_idx = np.where(bins == 2)[0]
        mid_mass = item_popularity[mid_idx].sum()

        print(f"  Total items: {n}, total mass: {total_mass}")
        print(f"  Target head mass r={r:.2f} → {target_mass_head:.0f}")
        print(f"  Bin1 (Short-head): {len(head_idx)} items, mass={head_mass} ({head_mass / total_mass * 100:.2f}%)")
        print(f"  Bin2 (Middle)    : {len(mid_idx)} items, mass={mid_mass} ({mid_mass / total_mass * 100:.2f}%)")
        print(f"  Bin3 (Long-tail) : {len(tail_idx)} items, mass={tail_mass} ({tail_mass / total_mass * 100:.2f}%)")

        self.stats['popularity_bins_mass'] = {
            'short_head_ratio': r,
            'bin1_mass': float(head_mass), 'bin2_mass': float(mid_mass), 'bin3_mass': float(tail_mass),
            'bin1_mass_pct': float(head_mass / total_mass), 'bin2_mass_pct': float(mid_mass / total_mass),
            'bin3_mass_pct': float(tail_mass / total_mass),
            'bin1_count': int(len(head_idx)), 'bin2_count': int(len(mid_idx)), 'bin3_count': int(len(tail_idx)),
            'labels': {'bin1': 'Short-head', 'bin2': 'Middle', 'bin3': 'Long-tail'}
        }
        self.popularity_bins_mass = bins
        self.item_popularity = item_popularity
        return bins, item_popularity, {'bin1': head_idx, 'bin2': mid_idx, 'bin3': tail_idx}

    def build_user_histories(self):
        """
        Build per-user train histories from self.train_data:
          - list-of-lists: indices of interacted items per user
        """
        csr = self.interaction_list_to_matrix().tocsr()
        self.user_hist_matrix = csr

        # list-of-lists
        self.user_histories = [csr[u].indices.astype(np.int64) for u in range(self.n_users)]
        return self.user_histories, self.user_hist_matrix

    def generate_user_groups(self, method='auto'):
        """
        Generate user groups based on dataset type and available files
        
        Args:
            method: 'auto', 'gender', 'activity'
        """
        print("\n" + "-"*80)
        print("Creating User Groups")
        print("-"*80)
        
        if method == 'auto':
            if 'ml-1m' in self.dataset_name or 'ml1m' in self.dataset_name:
                users_file = os.path.join(self.data_root, 'users.dat')
                if os.path.exists(users_file):
                    print(f"  Detected ML-1M dataset")
                    self.user_groups = self.create_user_groups_from_gender_ml1m(users_file)
                    return self.user_groups
                else:
                    print(f"  Warning: users.dat not found at {users_file}")
            
            elif 'nyc' in self.dataset_name or 'new-york' in self.dataset_name:
                # Try FTKY NYC profile file
                profile_file = os.path.join(self.data_root, 'dataset_UbiComp2016_UserProfile_NYC.txt')
                if os.path.exists(profile_file):
                    print(f"  Detected FTKY-NYC dataset")
                    self.user_groups = self.create_user_groups_from_gender_ftky(profile_file)
                    return self.user_groups
                else:
                    print(f"  Warning: NYC user profile not found at {profile_file}")
            
            elif 'tky' in self.dataset_name or 'tokyo' in self.dataset_name:
                # Try FTKY Tokyo profile file
                profile_file = os.path.join(self.data_root, 'dataset_UbiComp2016_UserProfile_TKY.txt')
                if os.path.exists(profile_file):
                    print(f"  Detected FTKY-TKY dataset")
                    self.user_groups = self.create_user_groups_from_gender_ftky(profile_file)
                    return self.user_groups
                else:
                    print(f"  Warning: TKY user profile not found at {profile_file}")
        elif method == 'gender':
            # Force gender method
            if 'ml-1m' in self.dataset_name or 'ml1m' in self.dataset_name:
                users_file = os.path.join(self.data_root, 'users.dat')
                if os.path.exists(users_file):
                    self.user_groups = self.create_user_groups_from_gender_ml1m(users_file)
                else:
                    raise FileNotFoundError(f"users.dat not found at {users_file}")
            elif 'nyc' in self.dataset_name:
                profile_file = os.path.join(self.data_root, 'dataset_UbiComp2016_UserProfile_NYC.txt')
                if os.path.exists(profile_file):
                    self.user_groups = self.create_user_groups_from_gender_ftky(profile_file)
                else:
                    raise FileNotFoundError(f"NYC profile not found at {profile_file}")
            elif 'tky' in self.dataset_name or 'tokyo' in self.dataset_name:
                profile_file = os.path.join(self.data_root, 'dataset_UbiComp2016_UserProfile_TKY.txt')
                if os.path.exists(profile_file):
                    self.user_groups = self.create_user_groups_from_gender_ftky(profile_file)
                else:
                    raise FileNotFoundError(f"TKY profile not found at {profile_file}")
            else:
                raise ValueError(f"Gender method not supported for dataset: {self.dataset_name}")
        
        elif method == 'activity':
            self.user_groups = self.create_user_groups_from_activity()
        
        return self.user_groups
    
    def save_fairness_data(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
        if self.item_groups is not None:
            item_groups_path = os.path.join(save_dir, 'item_groups.npy')
            np.save(item_groups_path, self.item_groups)
            print(f"\n  Saved: item_groups.npy")
        
        if self.user_groups is not None:
            user_groups_path = os.path.join(save_dir, 'user_groups.npy')
            np.save(user_groups_path, self.user_groups)
            print(f"  Saved: user_groups.npy")

        if hasattr(self, 'popularity_bins_mass'):
            bins_path = os.path.join(save_dir, 'popularity_bins_mass.npy')
            np.save(bins_path, self.popularity_bins_mass)
            print(f"  Saved: popularity_bins_mass.npy")

        if hasattr(self, 'item_popularity'):
            pop_path = os.path.join(save_dir, 'item_popularity.npy')
            np.save(pop_path, self.item_popularity)
            print(f"  Saved: item_popularity.npy")

        if self.user_hist_matrix is not None:
            hist_path = os.path.join(save_dir, 'hist_train.npz')
            sp.save_npz(hist_path, self.user_hist_matrix)
            print(f"  Saved: hist_train.npz")

        if self.user_histories is not None:
            uhist_path = os.path.join(save_dir, 'user_histories_train.npy')
            np.save(uhist_path, np.array(self.user_histories, dtype=object), allow_pickle=True)
            print(f"  Saved: user_histories_train.npy")

        info_path = os.path.join(save_dir, 'fairness_info.txt')
        with open(info_path, 'w') as f:
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("===== Data Dimensions =====\n")
            f.write(f"Num users: {self.n_users}\n")
            f.write(f"Num items: {self.n_items}\n\n")
            
            if 'item_groups' in self.stats:
                ig = self.stats['item_groups']
                f.write("===== Item Groups (Popularity-based) =====\n")
                f.write(f"Short-head ratio: {ig['short_head_ratio']*100:.1f}%\n")
                f.write(f"Short-head items (group 1): {ig['group_1_count']} items\n")
                f.write(f"Long-tail items (group 2): {ig['group_2_count']} items\n")
                f.write(f"Short-head avg interactions: {ig['short_head_avg_interactions']:.2f}\n")
                f.write(f"Long-tail avg interactions: {ig['long_tail_avg_interactions']:.2f}\n\n")

            if 'gini' in self.stats:
                gg = self.stats['gini']
                f.write("===== Gini (Train URM) =====\n")
                f.write(f"Gini_i (item interaction inequality): {gg['gini_i']:.6f}\n")
                f.write(f"Gini_u (user interaction inequality): {gg['gini_u']:.6f}\n\n")

            if 'gini_full' in self.stats:
                gf = self.stats['gini_full']
                f.write("===== Gini (Full Dataset: train+valid+test) =====\n")
                f.write(f"Total interactions: {gf['num_interactions']}\n")
                f.write(f"Gini_i (item interaction inequality): {gf['gini_i']:.6f}\n")
                f.write(f"Gini_u (user interaction inequality): {gf['gini_u']:.6f}\n\n")

            if 'user_groups' in self.stats:
                ug = self.stats['user_groups']
                f.write("===== User Groups =====\n")
                f.write(f"Method: {ug['method']}\n")
                if 'source_file' in ug:
                    f.write(f"Source file: {ug['source_file']}\n")
                f.write(f"{ug['group_1_label']} users (group 1): {ug['group_1_count']} ({ug['group_1_count']/self.n_users*100:.1f}%)\n")
                f.write(f"{ug['group_2_label']} users (group 2): {ug['group_2_count']} ({ug['group_2_count']/self.n_users*100:.1f}%)\n")
                if 'missing_count' in ug and ug['missing_count'] > 0:
                    f.write(f"Missing gender: {ug['missing_count']} ({ug['missing_count']/self.n_users*100:.1f}%)\n")
                f.write("\n")

            if 'popularity_bins_mass' in self.stats:
                pb = self.stats['popularity_bins_mass']
                f.write("===== Popularity Bins by MASS =====\n")
                f.write(f"Short-head mass ratio r: {pb['short_head_ratio']:.2f}\n")
                f.write(f"Bin1 (Short-head): count={pb['bin1_count']}, mass%={pb['bin1_mass_pct']:.2%}\n")
                f.write(f"Bin2 (Middle)    : count={pb['bin2_count']}, mass%={pb['bin2_mass_pct']:.2%}\n")
                f.write(f"Bin3 (Long-tail) : count={pb['bin3_count']}, mass%={pb['bin3_mass_pct']:.2%}\n\n")

            f.write("===== Files Generated =====\n")
            if self.item_groups is not None:
                f.write(f"item_groups.npy: ({self.n_items},) dtype=int64\n")
            if self.user_groups is not None:
                f.write(f"user_groups.npy: ({self.n_users},) dtype=int64\n")
        
        print(f"  Saved: fairness_info.txt")
        
        summary = {
            'dataset': self.dataset_name,
            'timestamp': datetime.now().isoformat(),
            'dimensions': {
                'n_users': self.n_users,
                'n_items': self.n_items
            }
        }
        
        if 'item_groups' in self.stats:
            summary['item_groups'] = self.stats['item_groups']
        if 'gini' in self.stats:
            summary['gini'] = self.stats['gini']
        if 'gini_full' in self.stats:
            summary['gini_full'] = self.stats['gini_full']

        if 'user_groups' in self.stats:
            summary['user_groups'] = self.stats['user_groups']
        if 'popularity_bins_mass' in self.stats:
            summary['popularity_bins_mass'] = self.stats['popularity_bins_mass']

        summary['files'] = {}
        if self.item_groups is not None:
            summary['files']['item_groups'] = 'item_groups.npy'
        if self.user_groups is not None:
            summary['files']['user_groups'] = 'user_groups.npy'
        if hasattr(self, 'popularity_bins_mass'):
            summary['files']['popularity_bins_mass'] = 'popularity_bins_mass.npy'
        if hasattr(self, 'item_popularity'):
            summary['files']['item_popularity'] = 'item_popularity.npy'

        if self.user_hist_matrix is not None:
            summary['files']['hist_train'] = 'hist_train.npz'
        if self.user_histories is not None:
            summary['files']['user_histories_train'] = 'user_histories_train.npy'

        json_path = os.path.join(save_dir, 'fairness_summary.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  Saved: fairness_summary.json")
    
    def generate_and_save(self, save_dir, user_group_method='auto',
                          valid_data=None, test_data=None):
        self.create_item_groups_from_popularity()
        self.generate_user_groups(method=user_group_method)
        self.create_popularity_bins(short_head_ratio=self.short_head_ratio)
        self.build_user_histories()
        self.compute_gini_iu()
        self.compute_gini_full(valid_data=valid_data, test_data=test_data)
        self.save_fairness_data(save_dir)
        
        self.verify_fairness_data(save_dir)
    
    def verify_fairness_data(self, fairness_dir):
        print("\n" + "-"*80)
        print("Verifying Fairness Data")
        print("-"*80)
        
        try:
            if os.path.exists(os.path.join(fairness_dir, 'user_groups.npy')):
                user_groups = np.load(os.path.join(fairness_dir, 'user_groups.npy'))
                assert len(user_groups) == self.n_users, f"user_groups size mismatch: {len(user_groups)} != {self.n_users}"
                assert set(user_groups).issubset({0,1,2}), f"Invalid user groups: {set(user_groups)}"
                print("  user_groups.npy verified")
            
            if os.path.exists(os.path.join(fairness_dir, 'item_groups.npy')):
                item_groups = np.load(os.path.join(fairness_dir, 'item_groups.npy'))
                assert len(item_groups) == self.n_items, f"item_groups size mismatch: {len(item_groups)} != {self.n_items}"
                assert set(item_groups).issubset({1,2}), f"Invalid item groups: {set(item_groups)}"
                print("  item_groups.npy verified")

            if os.path.exists(os.path.join(fairness_dir, 'popularity_bins_mass.npy')):
                bins = np.load(os.path.join(fairness_dir, 'popularity_bins_mass.npy'))
                assert len(bins) == self.n_items, f"popularity_bins_mass size mismatch: {len(bins)} != {self.n_items}"
                assert set(np.unique(bins)).issubset(
                    {1, 2, 3}), f"Invalid popularity bins labels: {set(np.unique(bins))}"
                print("  popularity_bins_mass.npy verified")

            print("  Fairness data verification passed")
        except AssertionError as e:
            print(f"  Verification failed: {e}")
            raise


class PreProcess:
    def __init__(self, args, dir_path, use_cache=False):
        self.dataset_name = args.dataset_name
        clean_dataset_split_path = os.path.join(dir_path,
                                       f'Clean_C{args.drop_num}_{args.split_ratio}.pt'.replace(" ", ""))

        if use_cache and os.path.exists(clean_dataset_split_path):
            print('Load cache datas')
            clean_dataset_info = torch.load(clean_dataset_split_path)

            self.sp_train = clean_dataset_info['train']
            self.sp_valid = clean_dataset_info['valid']
            self.sp_test = clean_dataset_info['test']
            self.num_users = clean_dataset_info['num_users']
            self.num_items = clean_dataset_info['num_items']

            self.df_user_pref_train = clean_dataset_info['user_pref_train']
            self.df_user_pref_train_valid = clean_dataset_info['user_pref_train_valid']
            self.df_user_pref_valid = clean_dataset_info['user_pref_valid']
            self.df_user_pref_test = clean_dataset_info['user_pref_test']
            self.num_cate = clean_dataset_info['num_cate']
            self.density = clean_dataset_info['density']
            self.num_interaction = clean_dataset_info['num_interaction']
            self.matrix_F = clean_dataset_info['matrix_F']
            self.item_category = clean_dataset_info['item_category']

            print('Done')
            print(
                f'num user: {self.num_users},',
                f'num item: {self.num_items},',
                f'num cate: {self.num_cate},',
                f'interaction: {self.num_interaction},',
                f'density: {self.density:.4f}')
        else:
            # set column name of data frame
            self.str_user, self.str_item, self.str_rating, self.str_time = args.str_cols
            clean_df_path = os.path.join(dir_path, f'clean_df_C{args.drop_num}.pt')

            file_path = os.path.join(dir_path, f'{args.file_name}')
            print('Read interaction datas')
            df = pd.read_csv(file_path, sep=args.sep, names=args.str_cols)
            print('Done')

            le_user = LabelEncoder() 
            le_item = LabelEncoder()
            print('Make clean datasets')
            # Drop and Sort interactions chronologically.
            df_clean = self.clean_and_sort(df, args.drop_num, args.drop_rating, le_user, le_item)
            print('Done')

            print('Store clean dataframe as clean_df.pt')
            df_dict = {
                'clean_df': df_clean,
                'user_enc': le_user,
                'item_enc': le_item,
            }
            torch.save(df_dict, clean_df_path)
            print('Done')

            print(f'Split datasets as train, valid, test: {args.split_ratio}')
            df_train, df_valid, df_test = self.split_group_by_user(df_clean, args.split_ratio, args.str_cols)
            print('Done')

            self.num_users = df_clean[self.str_user].nunique()
            self.num_items = df_clean[self.str_item].nunique()
            self.density = df_clean.shape[0] / (self.num_users * self.num_items)# density is the ratio of interactions to the total number of possible interactions
            self.num_interaction = df_clean.shape[0]

            print(
                f'num user: {self.num_users},',
                f'num item: {self.num_items},',
                # f'num cate: {self.num_cate},',
                f'interaction: {self.num_interaction},',
                f'density: {self.density:.4f}')
            print(f'Train: {len(df_train)}, Valid: {len(df_valid)}, Test: {len(df_test)}')
            train_list = df_train[[self.str_user, self.str_item]].values.astype(int)
            valid_list = df_valid[[self.str_user, self.str_item]].values.astype(int)
            test_list = df_test[[self.str_user, self.str_item]].values.astype(int)

            save_dir = os.path.join(dir_path, f"C{args.drop_num}_{args.split_ratio}".replace(" ", ""))
            os.makedirs(save_dir, exist_ok=True)

            np.save(os.path.join(save_dir, 'train_list.npy'), train_list)
            np.save(os.path.join(save_dir, 'valid_list.npy'), valid_list)
            np.save(os.path.join(save_dir, 'test_list.npy'), test_list)

            print(f"Saved npy files to: {save_dir}")
            print(f"  train_list: {train_list.shape}, valid_list: {valid_list.shape}, test_list: {test_list.shape}")
            
            if hasattr(args, 'create_recbole') and args.create_recbole:
                print("Creating RecBole atomic files...")
                recbole_dataset_dir = create_recbole_atomic_files(df_train, df_valid, df_test, self.dataset_name, save_dir)
                print(f"RecBole atomic files created in: {recbole_dataset_dir}")
                print(f"Directory structure: {os.path.dirname(recbole_dataset_dir)}")
            
            if hasattr(args, 'enable_fairness') and args.enable_fairness:
                print("\n" + "="*80)
                print("Generating Fairness Data")
                print("="*80)
                
                fairness_dir = os.path.join(save_dir, 'fairness')
                user_enc = df_dict['user_enc'] if os.path.exists(clean_df_path) else le_user
                original_user_ids = None
                if ('nyc' in self.dataset_name.lower()) or ('tky' in self.dataset_name.lower()):
                    original_user_ids = user_enc.classes_.astype(int)
                try:
                    fairness_gen = FairnessDataGenerator(
                        dataset_name=self.dataset_name,
                        n_users=self.num_users,
                        n_items=self.num_items,
                        train_data=train_list,
                        data_root=dir_path,
                        short_head_ratio=args.short_head_ratio if hasattr(args, 'short_head_ratio') else 0.2,
                        original_user_ids=original_user_ids
                    )
                    
                    user_method = args.fairness_user_method if hasattr(args, 'fairness_user_method') else 'auto'
                    fairness_gen.generate_and_save(fairness_dir, user_group_method=user_method,
                                                   valid_data=valid_list, test_data=test_list)
                    
                    print("\n" + "="*80)
                    print(f"✓ Fairness data successfully saved to: {fairness_dir}")
                    print("="*80)
                except Exception as e:
                    print(f"\n✗ Error generating fairness data: {e}")
                    print("Skipping fairness data generation...")
        
        summary_path = os.path.join(save_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Drop_num (k-core): {args.drop_num}\n")
            f.write(f"Drop_rating threshold: {args.drop_rating}\n")
            f.write(f"Split ratio (train/valid/test): {args.split_ratio}\n")
            f.write(f"Random seed: {args.seed}\n")
            f.write(f"CSV file: {args.file_name}\n")
            f.write(f"Separator: '{args.sep}'\n\n")
            f.write("===== Dataset Statistics =====\n")
            f.write(f"Num users: {self.num_users}\n")
            f.write(f"Num items: {self.num_items}\n")
            f.write(f"Num interactions: {self.num_interaction}\n")
            f.write(f"Density: {self.density:.6f}\n")
            f.write(f"Sparsity: {1 - self.density:.6f}\n\n")
            f.write("===== Split Sizes =====\n")
            f.write(f"Train interactions: {len(df_train)}\n")
            f.write(f"Valid interactions: {len(df_valid)}\n")
            f.write(f"Test interactions:  {len(df_test)}\n")

            if hasattr(args, 'enable_fairness') and args.enable_fairness:
                f.write("\n===== Fairness Data =====\n")
                fairness_dir = os.path.join(save_dir, 'fairness')
                if os.path.exists(fairness_dir):
                    f.write(f"Fairness data: GENERATED\n")
                    f.write(f"Location: {os.path.join('fairness', '')}\n")
                    if os.path.exists(os.path.join(fairness_dir, 'item_groups.npy')):
                        f.write(f"  - item_groups.npy\n")
                    if os.path.exists(os.path.join(fairness_dir, 'user_groups.npy')):
                        f.write(f"  - user_groups.npy\n")
                    if os.path.exists(os.path.join(fairness_dir, 'hist_train.npz')):
                        f.write(f"  - hist_train.npz\n")
                    if os.path.exists(os.path.join(fairness_dir, 'user_histories_train.npy')):
                        f.write(f"  - user_histories_train.npy\n")

                    if os.path.exists(os.path.join(fairness_dir, 'fairness_info.txt')):
                        f.write(f"  - fairness_info.txt\n")
                else:
                    f.write(f"Fairness data: NOT GENERATED\n")
        print(f"Summary saved to: {summary_path}")
        print(f'Data ready')

    def clean_and_sort(self, df, drop_num, drop_rating, le_user, le_item):
        def drop_unreliable(df, drop_rating):
            print('  Drop unreliable interaction')
            df_clean = df[df[self.str_rating] >= drop_rating]
            return df_clean

        def drop_unactive(df, str_col, drop_num):
            if str_col == self.str_user:
                df_group_size = df.groupby([self.str_user]).size()
            else:
                df_group_size = df.groupby([self.str_item]).size()

            clean_idx = df_group_size[df_group_size >= drop_num].index
            df_clean = df[df[str_col].isin(clean_idx)]
            return df_clean

        def is_unactive(df, str_col, drop_num):

            if str_col == self.str_user:
                df_group_size = df.groupby([self.str_user]).size()
            else:
                df_group_size = df.groupby([self.str_item]).size()

            unactive_df = df_group_size[df_group_size < drop_num].index
            print(f"    # of unactive interactions ({str_col}): {len(unactive_df)}")
            if len(unactive_df) == 0:
                return False  # False if unactive is None
            else:
                return True

        def core_setting(df, drop_num):
            print('  Drop unactive users and items.')

            while True:
                unactive_items = is_unactive(df, self.str_item, drop_num)
                unactive_users = is_unactive(df, self.str_user, drop_num)
                if not unactive_items and not unactive_users:
                    break
                if unactive_items:
                    df = drop_unactive(df, self.str_item, drop_num)
                if unactive_users:
                    df = drop_unactive(df, self.str_user, drop_num)

            return df

        # Drop duplicated (user, tiem)
        df = df.drop_duplicates([self.str_user, self.str_item]).reset_index(drop=True)
        if drop_rating != -1:
            df = drop_unreliable(df, drop_rating)
        if drop_num:
            df = core_setting(df, drop_num)
        if self.dataset_name=='ml-1m':
            df[self.str_rating] = 1
        # sort chronologically
        df_sorted = df.sort_values([self.str_user, self.str_time])

        # Encode user/item id
        df_sorted[self.str_user] = le_user.fit_transform(df_sorted[self.str_user])
        df_sorted[self.str_item] = le_item.fit_transform(df_sorted[self.str_item])
        return df_sorted

    def split_group_by_user(self, df, ratio, str_cols):
        np_data = df.values
        group_users = df.groupby(self.str_user)

        sum_ratio = sum(ratio)
        test_ratio = round(ratio[2] / sum_ratio, 3)
        sum_ratio -= ratio[2]
        val_ratio = round(ratio[1] / sum_ratio, 3)

        num_cum_items = 0

        train = []
        valid = []
        test = []

        for user_id, df_user in tqdm(group_users):
            num_items = len(df_user)
            num_test = np.ceil(num_items * test_ratio).astype(int)  
            num_valid = np.ceil((num_items - num_test) * val_ratio).astype(int)
            num_train = int(num_items - num_test - num_valid)

            train.extend(np_data[num_cum_items:num_cum_items + num_train, :])
            valid.extend(np_data[num_cum_items + num_train:num_cum_items + num_train + num_valid, :])
            test.extend(
                np_data[num_cum_items + num_train + num_valid:num_cum_items + num_train + num_valid + num_test, :])
            num_cum_items += num_items

        np_train = np.array(train)
        np_valid = np.array(valid)
        np_test = np.array(test)

        df_train = pd.DataFrame(data=np_train, columns=str_cols)
        df_valid = pd.DataFrame(data=np_valid, columns=str_cols)
        df_test = pd.DataFrame(data=np_test, columns=str_cols)

        return df_train, df_valid, df_test

def create_recbole_atomic_files(df_train, df_valid, df_test, dataset_name, save_dir):
    """
    Create RecBole atomic files from train/valid/test dataframes.
    
    Args:
        df_train: Training dataframe with columns [user, item, rating, timestamp]
        df_valid: Validation dataframe with columns [user, item, rating, timestamp]
        df_test: Test dataframe with columns [user, item, rating, timestamp]
        dataset_name: Name of the dataset
        save_dir: Directory to save the atomic files (e.g., C5_[6,2,2])
    """
    import os
    
    # Create recbole directory structure: save_dir/recbole/dataset_name/
    # This will create: C5_[7,1,2]/recbole/ml-1m/
    recbole_dir = os.path.join(save_dir, 'recbole')
    dataset_dir = os.path.join(recbole_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    df_all = pd.concat([df_train, df_valid, df_test], ignore_index=True)
    
    user_df = df_all[['user']].drop_duplicates().reset_index(drop=True)
    user_df['user_id:token'] = user_df['user']
    user_df = user_df[['user_id:token']]
    user_file = os.path.join(dataset_dir, f'{dataset_name}.user')
    user_df.to_csv(user_file, sep='\t', index=False)
    print(f"Created user file: {user_file}")
    
    # Create item atomic file
    item_df = df_all[['item']].drop_duplicates().reset_index(drop=True)
    item_df['item_id:token'] = item_df['item']
    item_df = item_df[['item_id:token']]
    item_file = os.path.join(dataset_dir, f'{dataset_name}.item')
    item_df.to_csv(item_file, sep='\t', index=False)
    print(f"Created item file: {item_file}")
    
    # Create interaction atomic files
    def create_interaction_file(df, split_name):
        inter_df = df.copy()
        inter_df['user_id:token'] = inter_df['user']
        inter_df['item_id:token'] = inter_df['item']
        inter_df['timestamp:float'] = inter_df['timestamp']
        inter_df = inter_df[['user_id:token', 'item_id:token', 'timestamp:float']]
        

        filename = f'{dataset_name}.{split_name}.inter'
        
        file_path = os.path.join(dataset_dir, filename)
        inter_recbole = inter_df.astype(int)
        inter_recbole.to_csv(file_path, sep='\t', index=False)
        print(f"Created {split_name} interaction file: {file_path}")
        return file_path
    
    # Create train, valid, test interaction files
    
    create_interaction_file(df_train, 'train')
    create_interaction_file(df_valid, 'valid')
    create_interaction_file(df_test, 'test')
    
    # Create info.txt file
    info_file = os.path.join(dataset_dir, 'info.txt')
    with open(info_file, 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Num users: {len(user_df)}\n")
        f.write(f"Num items: {len(item_df)}\n")
        f.write(f"Num interactions: {len(df_all)}\n")
        f.write(f"Train interactions: {len(df_train)}\n")
        f.write(f"Valid interactions: {len(df_valid)}\n")
        f.write(f"Test interactions: {len(df_test)}\n")
    
    print(f"Created info file: {info_file}")
    print(f"RecBole atomic files created in: {dataset_dir}")
    print(f"RecBole directory structure: {recbole_dir}")
    
    return dataset_dir


def set_random_seed(random_seed):
    torch.manual_seed(random_seed)  
    torch.cuda.manual_seed(random_seed)  
    np.random.seed(random_seed)  
    random.seed(random_seed)  
    torch.backends.cudnn.deterministic = True  


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Preprocessing", add_help=True)

    parser.add_argument('--seed', default=42, type=int, help="Random seed")
    parser.add_argument('--dataset_name', default='ftky', type=str, help="Dataset name")
    parser.add_argument('--data_path', default="", type=str, help="Path to dataset directory. If not specified, uses dataset/{dataset_name}")
    parser.add_argument('--str_cols', default=['user', 'item', 'rating', 'timestamp'],
                        type=str, nargs="+", help="Interaction dataframe column names")
    parser.add_argument('--file_name', default='data.csv', type=str, help="Interaction file name")
    parser.add_argument('--drop_num', default=20, type=int, help="Drop user whose history are less than drop_num")
    parser.add_argument('--drop_rating', default=-1, type=int,
                        help="Drop interaction of which rating is less than drop_rating")
    parser.add_argument('--split_ratio', default=[7, 1, 2], type=int, nargs="+", help="Train, Valid, Test split ratio")
    parser.add_argument('--sep', default=',', type=str, help="Seperator of interaction csv file")
    parser.add_argument('--create_recbole',action='store_true', help="Create RecBole atomic files")
    
    parser.add_argument('--enable_fairness', action='store_true', 
                        help="Generate fairness data (user_groups.npy, item_groups.npy)")
    parser.add_argument('--short_head_ratio', default=0.2, type=float,
                        help="Ratio of items to consider as short-head for item groups (default: 0.2)")
    parser.add_argument('--fairness_user_method', default='activity',
                        choices=['auto', 'gender', 'activity'],
                        help="Method to create user groups: auto (detect dataset), gender, or activity (default: auto)")

    args = parser.parse_args()
    set_random_seed(random_seed=args.seed)
    
    if args.data_path is not None:
        dir_path = os.path.join(args.data_path, args.dataset_name)
    else:
        dir_path = os.path.join(os.getcwd(), 'dataset', args.dataset_name)
    
    print(f'Preprocess {args.dataset_name}: Core-{args.drop_num} setting')
    print(f'Data path: {dir_path}')
    
    if args.enable_fairness:
        print(f'Fairness data generation: ENABLED')
        print(f'  - Short-head ratio: {args.short_head_ratio}')
        print(f'  - User group method: {args.fairness_user_method}')
    
    dataset = PreProcess(args, dir_path, use_cache=False)