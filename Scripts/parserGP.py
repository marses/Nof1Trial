#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 10:14:45 2019

@author: seslija
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:37:49 2019

@author: seslija

Testing the effects of Star on L1 metrics.
"""




import pandas
import numpy
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import integrate
from datetime import datetime
import sys
import math
import re

from SuppScript import inverse_quaternion, quaternion_multiply,\
                        quaternion_to_euler, euler_to_quaternion,\
                        detect_cusum, peaks, inflection_points,\
                        onset_peaks, number_of_flips

#import SuppScript
from CUMSUM_flip_detection import CUMSUM_flip, SampEn

plt.close("all")
pandas.set_option('display.max_columns', None)

#                                                                                           1556104508_mb-UqD0jh23v-SrmESuT7o-E1VUYuWrfypdXWapYMXF
#df = pandas.read_csv("data/1553686387_server_instance_study_name_group_name_invite_code.csv")
#                                                                            /1556104508_-h1aD3lYSQpo4bhSfN4ex8UO7tzzq49E7MY2CKItWvS
plt.close("all")

#files_list = [[15,'OX015.csv'], [35,'OX035.csv'],\
#[51,'OX051.csv'], [3,'OX003.csv'],\
#[17,'OX017.csv'], [350,'OX0350.csv'],\
#[55,'OX055.csv'], [11,'OX011.csv'],\
#[23,'OX023.csv'], [36,'OX036.csv'],\
#[58,'OX058.csv'], [12,'OX012.csv'], \
#[26,'OX026.csv'], [38,'OX038.csv'],\
#[62,'OX062.csv'], [13,'OX013.csv'],\
#[30,'OX030.csv'], [39,'OX039.csv'],\
#[66,'OX066.csv']]

files_list = [[15,'OX015.csv'], [35,'OX035.csv'],\
[51,'OX051.csv'], [3,'OX003.csv'],\
[17,'OX017.csv'], [350,'OX0350.csv'],\
[55,'OX055.csv'], [11,'OX011.csv'],\
[23,'OX023.csv'], [36,'OX036.csv'],\
[58,'OX058.csv'], [12,'OX012.csv'], \
[38,'OX038.csv'],\
[62,'OX062.csv'], [13,'OX013.csv'],\
[39,'OX039.csv'],\
[66,'OX066.csv']]

for fileX in files_list[1:2]:#12:13]:
    ID = fileX[0]
    file_name = fileX[1]
    #                                                                                           1556104508_mb-UqD0jh23v-SrmESuT7o-E1VUYuWrfypdXWapYMXF
    #df = pandas.read_csv("data/1553686387_server_instance_study_name_group_name_invite_code.csv")
    #                                                                            /1556104508_-h1aD3lYSQpo4bhSfN4ex8UO7tzzq49E7MY2CKItWvS
    df = pandas.read_csv("/Users/seslija/Documents/GameChanger/Accelerometer/tiltraw/"+file_name,\
                         usecols = lambda column : column not in ['instanceId','gamePauseReason','tiltGyroCount', 'tiltMotionCount', 'gyroAt', 'gyroX', 'gyroY', 'gyroZ', 'motionPitch', 'motionRoll', 'motionYaw', 'motionRotX', 'motionRotY', 'motionRotZ', 'motionGrvX', 'motionGrvY', 'motionGrvZ', 'motionAccX', 'motionAccY', 'motionAccZ'],\
                         dtype = {'scheduleItemKeys' : 'object',\
                                  'gameState' : 'object',\
                                  'gameType' : 'object',\
                                  'at' : 'float64',\
                                  'atRaw' : 'float64'},\
                         verbose = 0)
    
    output_all = pandas.DataFrame()
    #1557315390_frWcZqO3TDcOAVpDtuVYxicYWuVRLjbo7ukNrgop9Kv.csv
    #df = df.iloc[0:500217]
    # remove columns
    #df = df.drop(columns=['tiltGyroCount', 'tiltMotionCount', 'gyroAt', 'gyroX', 'gyroY', 'gyroZ', 'motionPitch', 'motionRoll', 'motionYaw', 'motionRotX', 'motionRotY', 'motionRotZ', 'motionGrvX', 'motionGrvY', 'motionGrvZ', 'motionAccX', 'motionAccY', 'motionAccZ'])
    
    # drop duplicates
    df = df.drop_duplicates()
    
    
    scheduleStepBegan = df.loc[df['type'] == 'payloadType_scheduleStepBegan'][['at','atRaw']]
    
    scheduleItemKeys = df['scheduleItemKeys']#.dropna()
    #scheduleItemKeys_instance = scheduleItemKeys.str.split('-',n=4).tolist()
    #scheduleItemKeys_instance = [column[2] for column in scheduleItemKeys_instance]
    scheduleItemKeys_instance = scheduleItemKeys.str.split('-',expand=True)[2]#.dropna()
    scheduleItemKeys_instance_in_numbers = []
    scheduleItemKeys_instance_in_numbers_baseline = []
    
    for i in scheduleItemKeys_instance:
        if i =='Baseline.':
            scheduleItemKeys_instance_in_numbers.append(1)
        elif i == '2':
            scheduleItemKeys_instance_in_numbers.append(2)
        elif i == '3':
            scheduleItemKeys_instance_in_numbers.append(3)
        else:
            scheduleItemKeys_instance_in_numbers.append(float('nan'))
    # instance tells us if the event belongs to baseline, first or second follow-up
    df['instance'] = numpy.array(scheduleItemKeys_instance_in_numbers)#.astype(int)
    
    df_instance = df['instance'].values
    instances_all = numpy.zeros(df.shape[0])#.astype(int)
    instances_all[0] = df_instance[0]
    for i in range(0,len(instances_all)):
        if not(math.isnan(df_instance[i])):
            temp_instance = int(df_instance[i])
        if math.isnan(df_instance[i]):
            instances_all[i] = int(temp_instance)
    
    df['instance_'] = instances_all.astype(int)
    
    # instances belonging to Baseline
    instance_baseline = df.loc[df['instance']<2]['instance']
    instance_2 = df.loc[df['instance']==2]['instance']
    instance_3 = df.loc[df['instance']==3]['instance']
    
    
    # start and end times for eache instance of study
    #if not(instance_baseline.empty):
    #    baseline_start = df.loc[df['instance']<2]['at'].iloc[0]
    #    baseline_end = df.loc[df['instance']<2]['at'].iloc[-1]
    #if not(instance_2.empty):
    #    instance_2_start = df.loc[df['instance']==2]['at'].iloc[0]
    #    instance_2_end = df.loc[df['instance']==2]['at'].iloc[-1]
    #if not(instance_3.empty):
    #    instance_3_start = df.loc[df['instance']==3]['at'].iloc[0]
    #    instance_3_end = df.loc[df['instance']==3]['at'].iloc[-1]
    #
    #if not(instance_baseline.empty):
    #    set_schedule = numpy.zeros(instance_baseline.index.max()).astype(int)
    #    for j in range(0,len(instance_baseline.index)-1):
    #        set_schedule[instance_baseline.index[j]:instance_baseline.index[j+1]] = j
    #    set_schedule[instance_baseline.index[j+1]:len(set_schedule)] = j+1
    #    set_schedule_max = set_schedule.max()
    #else:
    #    # when baseline is absent in dataset
    #    set_schedule = numpy.array([])
    #    set_schedule_max = 0
        
    
    #set_ = numpy.zeros(df.shape[0]).astype(int)
    #set_[0:set_schedule.shape[0]] = set_schedule
    #for i in range(set_schedule_max+1,len(scheduleStepBegan.index)-1):
    ##for i in range(j+1,len(scheduleStepBegan.index)-1):
    #    set_[scheduleStepBegan.index[i]:scheduleStepBegan.index[i+1]]= i
    #set_[scheduleStepBegan.index[i+1]:len(set_)] = i+1
    #
    ## set_ is a new variable which identifies a game session (instance) number in consecutive order
    #df['set_'] = set_
    #
    #index_where_set_changes = numpy.append(0,numpy.where(set_[:-1] != set_[1:])[0] + 1)
    
    # 
    """
    #
    # This is another way to generate session
    #
    # identifies a session: 3x7 in one/two/three weeks of data
    session_ = numpy.zeros(set_.max()+1).astype(int)
    for i in range(0,set_.max()+1):#set_.max()):
        i_index = index_where_set_changes[i]
        i_index_minus_one = index_where_set_changes[i-1]
        #session_[i] = session_[i-1] + 1
        if (numpy.array(datetime.fromtimestamp(df['at'].loc[i_index]).day) == numpy.array(datetime.fromtimestamp(df['at'].loc[i_index_minus_one]).day)\
                          and ( (numpy.array(datetime.fromtimestamp(df['at'].loc[i_index]).hour) == numpy.array(datetime.fromtimestamp(df['at'].loc[i_index_minus_one]).hour))\
                             or (numpy.array(datetime.fromtimestamp(df['at'].loc[i_index]).hour) == numpy.array(datetime.fromtimestamp(df['at'].loc[i_index_minus_one]).hour + 1 )))):       
            session_[i] = session_[i-1]
        elif (numpy.array(datetime.fromtimestamp(df['at'].iloc[i_index]).day) == numpy.array(datetime.fromtimestamp(df['at'].iloc[i_index_minus_one]).day)\
                          and ( (numpy.array(datetime.fromtimestamp(df['at'].loc[i_index]).hour) == numpy.array(datetime.fromtimestamp(df['at'].loc[i_index_minus_one]).hour))\
                             or (numpy.array(datetime.fromtimestamp(df['at'].loc[i_index]).hour) == numpy.array(datetime.fromtimestamp(df['at'].loc[i_index_minus_one]).hour + 2 )))):       
            session_[i] = session_[i-1]
    
        else:
            session_[i] = session_[i-1] + 1
    session_ = session_-1
    
    
    session_all = numpy.zeros(df.shape[0]).astype(int)
    j = 0
    for i in range(0,df.shape[0]-1):
        if set_[i+1] == set_[i]:
           session_all[i+1] = session_[j]
        else:
            session_all[i+1] = session_all[i]
            if j < set_.max():
                j = j + 1        
    # make a new column which identifies a session by consecutive number
    df['session'] = session_all
                    
                
    """
    
    
    # since the state game is not directly reported in the Correct/Incorrect rows
    # we shall create a new column which will contain the name of the game in 
    # Correct/Incorrect row
    game_name_column = [None]*len(df)
    StimulusAppear_game_name = df.loc[df['type'] == 'payloadType_tiltStimulusAppear']['tiltTrialsName']
    
    index_current = StimulusAppear_game_name.index[:-1]
    index_next = StimulusAppear_game_name.index[1:]
    for i in range(0,len(index_current)):
        game_name_column[index_current[i]:index_next[i]] = [StimulusAppear_game_name[index_current[i]]]*(index_next[i]-index_current[i])
    
    
    
    
    
    
    time_at = df['at'].values
    ses = numpy.zeros(df.shape[0]).astype(int)
    for i in range(1,df.shape[0]):
        if not(numpy.isnan(time_at[i-1])):
            t_temp = time_at[i-1]
        if (time_at[i] - t_temp) > 3600:
            ses[i] = ses[i-1] + 1
        else:
            ses[i] = ses[i-1]
    df['ses'] = ses
    
    
    #"""
    # Quaternions
    quat = df[['motionAt','motionQuatX','motionQuatY','motionQuatZ','motionQuatW']]
    quat_raw = quat.dropna()
    #quatT = quat_raw.iloc[:,0]
    #quatX = quat_raw.iloc[:,1]
    #quatY = quat_raw.iloc[:,2]
    #quatZ = quat_raw.iloc[:,3]
    #quatW = quat_raw.iloc[:,4]
    #"""
    
    GameStarts = df.loc[df['type'] == 'payloadType_gameStateStart']['at']
    GameEnds = df.loc[df['type'] == 'payloadType_gameStateEnd']['at']
    total_duration_of_game = GameEnds.max() - GameStarts.min()
    scheduleStepResume = df.loc[df['type'] == 'payloadType_gameStateResume'][['at','atRaw','motionAt']]
    gameStatePause = df.loc[df['type'] == 'payloadType_gameStatePause'][['at','atRaw']]
    
    StimulusAppear = df.loc[df['type'] == 'payloadType_tiltStimulusAppear'][['at','atRaw','motionAt']]
    tiltCorrectIncorrect = df.loc[(df['type'] == 'payloadType_tiltCorrect') | (df['type'] == 'payloadType_tiltIncorrect')][['at','atRaw']]
    
    tiltCorrect = df.loc[(df['type'] == 'payloadType_tiltCorrect')][['at','atRaw','motionAt']]
    tiltIncorrect = df.loc[(df['type'] == 'payloadType_tiltIncorrect')][['at','atRaw','motionAt']]
    
    # taking care of reference - 28 March 
    payloadType_tiltMove_raw = df.loc[df['type'] == 'payloadType_tiltMove'][['tiltCentreAtt_x','tiltCentreAtt_y','tiltCentreAtt_z']]
    payloadType_tiltMove = payloadType_tiltMove_raw[1::2]
    # adding a missing column (w), which was not recorded
    #payloadType_tiltMove['tiltCentreAtt_w']=numpy.sqrt(1-numpy.sum(payloadType_tiltMove**2,1))
    df.iloc[:,df.columns.get_loc('tiltCentreAtt_w')] = numpy.sqrt(1-numpy.sum(payloadType_tiltMove**2,1))
    
    # identifying windows
    TiltCompleted = tiltCorrectIncorrect.values
    #StimulusAppeared_added = numpy.zeros(shape=(len(TiltCompleted)-len(StimulusAppear),))
    StimulusAppeared = StimulusAppear.values
    #StimulusAppeared = numpy.hstack([StimulusAppeared,StimulusAppeared_added])
    for i in range(0,len(TiltCompleted)):
        if StimulusAppeared.shape[0]< i: 
            if StimulusAppeared[i,1] > TiltCompleted[i,1]:
                StimulusAppeared = numpy.insert(StimulusAppeared, i, StimulusAppeared[i-1,:],axis=0)
                #StimulusAppeared = numpy.insert(StimulusAppeared, i, TiltCompleted[i-1,0],axis=0)
                #StimulusAppeared = numpy.insert(StimulusAppeared, i, StimulusAppeared[i-1])
        else:
            StimulusAppeared = numpy.insert(StimulusAppeared, i, StimulusAppeared[i-1,:],axis=0)
    
    
    output_all = pandas.DataFrame()
            
    suspicious_counter = 0        
    #troubleStarred
    #
    #game_name = 'numbersStarred'#'numbers'#'troubleStarred'#'reaction'#'troubleStarred'#'doubleTrouble'#'troubleStarred'#'reaction'# 'numbers'#'doubleTrouble' 'numbersStarred' 'troubleStarred'
#    #
#    avg = numpy.array([])
#    avg_all = numpy.array([])
#    std = numpy.array([])
#    
#    time_reaction_all = numpy.array([])
#    total_move = numpy.array([])
#    total_time = numpy.array([])
#    time_ = numpy.array([])
#    
#    tot_incorrect = []#range(154-14,154-13):#
#    
#    ## Record all times not just average
#    time_reaction_all_values = numpy.array([]).astype(float)
#    time_event_all_values = numpy.array([]).astype(float)
#    # Correct/Incorrect label
#    corrIncorr_label = numpy.array([]).astype(int)
#    # Star present or not
#    array_st_all = numpy.array([]).astype(int)
#    ##
#    L1_trigger_all = numpy.array([]).astype(float)
#    L1_remaining_all = numpy.array([]).astype(float)
#    SE_y_all = numpy.array([]).astype(float)
#    SE_z_all = numpy.array([]).astype(float)
#    #
#    # flip counter
#    flips_arr = numpy.array([]).astype(int)
#    # max level reched
#    level_arr = numpy.array([]).astype(int)
#    
#    # if not supplied, ref is
#    ref_quaternion = numpy.array([0,0,0,1])
#    Incorr_Total_array = []
#    
#    
#    # number of windows of game_name
#    number_of_windows = 0
#    
#    # all Flips
#    Flips_all = []
#    
#    # falling for the opposite side when Star present
#    fallen_for_opposite_temp_all = []
#    
#    # fallen for shape lure
#    fallen_for_lure_temp_all = []
#    
#    # time of the first move in session
#    time_session_all = []
    
    # amount of data considered & total data
    # number of windows under consideration
    total_data = 0
    considered_data = 0

    for game_name in ['numbersStarred']:#['reaction']:#, 'numbers', 'numbersStarred', 'doubleTrouble', 'troubleStarred']:
        plt.close("all")
        level_arr = numpy.array([]).astype(int)
        
        # --- Begin Restarting ---
        # Restart the arrays at the begining of a new game
        avg = numpy.array([])
        avg_all = numpy.array([])
        std = numpy.array([])
        
        time_reaction_all = numpy.array([])
        total_move = numpy.array([])
        total_time = numpy.array([])
        time_ = numpy.array([])
        
        tot_incorrect = []#range(154-14,154-13):#
        
        ## Record all times not just average
        time_reaction_all_values = numpy.array([]).astype(float)
        time_event_all_values = numpy.array([]).astype(float)
        # Correct/Incorrect label
        corrIncorr_label = numpy.array([]).astype(int)
        # Star present or not
        array_st_all = numpy.array([]).astype(int)
        ##
        L1_trigger_all = numpy.array([]).astype(float)
        L1_remaining_all = numpy.array([]).astype(float)
        SE_y_all = numpy.array([]).astype(float)
        SE_z_all = numpy.array([]).astype(float)
        #
        MD_trigger_all = numpy.array([]).astype(float)
        MD_remaining_all = numpy.array([]).astype(float)
        #
        # number of turns in angle signal
        flips_trigger_all = numpy.array([]).astype(float)
        flips_remaining_all = numpy.array([]).astype(float)
        
        #
        signal_trigger_all = numpy.array([]).astype(float)
        signal_remaining_all = numpy.array([]).astype(float)
       
        
        #
        # flip counter
        flips_arr = numpy.array([]).astype(int)
        # max level reched
        level_arr = numpy.array([]).astype(int)
        
        # if not supplied, ref is
        ref_quaternion = numpy.array([0,0,0,1])
        Incorr_Total_array = []
        
        
        # number of windows of game_name
        number_of_windows = 0
        
        # all Flips
        Flips_all = []
        
        # falling for the opposite side when Star present
        fallen_for_opposite_temp_all = []
        
        # fallen for shape lure
        fallen_for_lure_temp_all = []
        
        # time of the first move in session
        time_session_all = []
        
        # --- End Restarting ---
        

        Window_sessions = range(0,int(ses.max())+1)
        if ID == 38:
            list_w = [x_w for x_w in Window_sessions if x_w != 16]
        else:
            list_w = [x_w for x_w in Window_sessions]
        #list_w = [16]
        for k in list_w[0:5]:#
            window_i = k 
            #window = df.loc[df['set_' == window_i]]
            StimulusAppear_window = df.loc[((df['type'] == 'payloadType_tiltStimulusAppear') & (df['tiltTrialsName'] == 'tiltTrialName_' + game_name)) & (df['ses'] == window_i )][['at','atRaw','atFormatted','ses','motionAt','tiltTrialArr','type','tiltSlot']]
            tiltCorrectIncorrect_window = df.loc[(((df['type'] == 'payloadType_tiltCorrect') | (df['type'] == 'payloadType_tiltIncorrect'))) & (df['ses'] == window_i) ][['at','atRaw','atFormatted','ses','motionAt','tiltTrialArr','type']]
            
            Stimulus_and_CorrIncorr = df.loc[(((df['type'] == 'payloadType_tiltCorrect') | (df['type'] == 'payloadType_tiltIncorrect') | (df['type'] == 'payloadType_tiltStimulusAppear'))) & (df['ses'] == window_i) ][['at','atRaw','atFormatted','ses','motionAt','tiltTrialArr','type','tiltSlot']]
            
            # Moves
            Moves_window = df.loc[((df['type'] == 'payloadType_tiltMove')  & (df['ses'] == window_i ) )][['at','atRaw','atFormatted','ses','motionAt','tiltTrialArr','type','tiltSlot','tiltCursorState','tiltCursorTo']]
            fallen_for_lure_temp = []
            
         
            
            # print(tiltCorrectIncorrect_window)
            # restrict window to only those stimula followed by correct/incorrect answer
            # exclude those stimuli not associated with a response
            
            # May 9 - 21:40
            all_index = Stimulus_and_CorrIncorr.index.tolist()
            Stimulus_and_CorrIncorr_index = Stimulus_and_CorrIncorr.index.tolist()
            StimulusAppear_window_index = StimulusAppear_window.index.tolist()
            tiltCorrectIncorrect_window_index = tiltCorrectIncorrect_window.index.tolist()
            
            #stimApp = [x for x in all_index if x in StimulusAppear_window_index]
            #corIncorr_index = [y for y in all_index if y in tiltCorrectIncorrect_window_index]
            
            # A, B, C are indexes which identify next/previous/current nonempty row in 'at' column
            
            selection = []
            for i in range(0,len(StimulusAppear_window)):
                index_i = all_index.index(StimulusAppear_window.index[i])
                #index_corrIncorr = all_index.index(tiltCorrectIncorrect_window.index[i])
                #selection.append(StimulusAppear_window.index[i] in tiltCorrectIncorrect_window.index[index_i-1])
                index_corrIncorr = numpy.where(Stimulus_and_CorrIncorr.index == StimulusAppear_window.index[i])[0][0]
                B = [numpy.where(Stimulus_and_CorrIncorr.index == i)[0][0] for i in tiltCorrectIncorrect_window.index.tolist()]
                #index_corrIncorr1 = numpy.where(Stimulus_and_CorrIncorr.index[i] == StimulusAppear_window.index[i])[0][0]
                #A = Stimulus_and_CorrIncorr.index[index_corrIncorr + 1]
                #tiltCorrectIncorrect_window.index-1
                selection.append(StimulusAppear_window.index[i] in Stimulus_and_CorrIncorr.index[[x-1 for x in B]].tolist())
            
            
            only_these_stimuli = [i for i, x in enumerate(selection) if x]
            StimulusAppear_window_restricted = StimulusAppear_window.iloc[only_these_stimuli]
            
            # eliminate those that contain NaN at 'tiltTrialArr'
            StimulusAppear_window_restricted = StimulusAppear_window_restricted[pandas.notnull(StimulusAppear_window_restricted['tiltTrialArr'])]
            
            
            # compute the amount of discarded data
            total_data = total_data + len(StimulusAppear_window)
            #considered_data = considered_data + len(StimulusAppear_window_restricted)
            
            selection_move = []
            for i in range(0,len(tiltCorrectIncorrect_window)):
                C = [numpy.where(Stimulus_and_CorrIncorr.index == i)[0][0] for i in StimulusAppear_window_restricted.index.tolist()]
                selection_move.append(tiltCorrectIncorrect_window.index[i] in Stimulus_and_CorrIncorr.index[[x+1 for x in C]].tolist())
                                      #StimulusAppear_window_restricted.index+1)
            
            only_these_moves = [i for i, x in enumerate(selection_move) if x]
            #tiltCorrectIncorrect_window_restricted = tiltCorrectIncorrect_window.iloc[only_these_moves]
            tiltCorrectIncorrect_window_restricted = tiltCorrectIncorrect_window.iloc[only_these_moves]
            
            # eliminate those windows than are not in the interval (0.2,10) seconds
            windowTime = tiltCorrectIncorrect_window_restricted['atRaw'].values - StimulusAppear_window_restricted['atRaw'].values
            windows_array = []
            for i in range(0,len(windowTime)):
                if (windowTime[i] > 0.2 and windowTime[i] < 10):
                    windows_array.append(i)
                    
            tiltCorrectIncorrect_window_restricted = tiltCorrectIncorrect_window_restricted.iloc[windows_array]
            StimulusAppear_window_restricted = StimulusAppear_window_restricted.iloc[windows_array]
            
            temporal_arr_st = []
            temporary_label = []
            if len(tiltCorrectIncorrect_window_restricted) > 0:        
                #
                # Detecting the flips in stimuli
                #
                Trial = StimulusAppear_window_restricted[['tiltTrialArr','tiltSlot']]
        
                letter_or_number = numpy.zeros(len(Trial),dtype=int)
                for i_slot in range(0,len(Trial['tiltSlot'])):
                    String = Trial['tiltTrialArr'].iloc[i_slot]
                    if isinstance(Trial['tiltTrialArr'].iloc[i_slot], str):
                        ReLis = re.findall('(\w+)', String)
                        for word in ReLis:
                            TempWordx = re.findall('x$',word)
                            if TempWordx != []:
                                # 1 is letter, 0 if number is the target
                                letter_or_number[i_slot] = word[-2].isalpha()
                
                number_of_windows += 1                
                                     
                                
                            
                StimulusAppear_window_restricted_slot_0 =\
                                StimulusAppear_window_restricted.loc[
                                        StimulusAppear_window_restricted['tiltSlot'] 
                                        == StimulusAppear_window_restricted['tiltSlot'].iloc[0]]
                                
                StimulusAppear_window_restricted_slot_1 =\
                                StimulusAppear_window_restricted.loc[
                                        StimulusAppear_window_restricted['tiltSlot'] 
                                        == StimulusAppear_window_restricted['tiltSlot'].iloc[0]+1]
                
                if len(StimulusAppear_window_restricted_slot_1) == 0:
                    StimulusAppear_window_restricted_slot_1 =\
                                StimulusAppear_window_restricted.loc[
                                        StimulusAppear_window_restricted['tiltSlot'] 
                                        != StimulusAppear_window_restricted['tiltSlot'].iloc[0]]
        
                        
                Flip_0 = numpy.zeros(len(StimulusAppear_window_restricted_slot_0), dtype = int)
                Flip_1 = numpy.zeros(len(StimulusAppear_window_restricted_slot_1), dtype = int)
                
                letter_or_number_0 = letter_or_number[0:len(Flip_0)]
                letter_or_number_1 = letter_or_number[len(Flip_0):]
                
                index_of_change_0 = numpy.where(numpy.diff(numpy.sign(letter_or_number_0)))[0] + 1  
                index_of_change_1 = numpy.where(numpy.diff(numpy.sign(letter_or_number_1)))[0] + 1
                Flip_0[index_of_change_0] = 1
                if len(Flip_1)> 0:
                    Flip_1[index_of_change_1] = 1
                Flips = numpy.concatenate((Flip_0,Flip_1))
                
                Flips_all = numpy.append(Flips_all,Flips)
                
                # max level achieved
                level_0 = len(Flip_0)
                level_1 = len(Flip_1)
                level_max = max(level_0,level_1)
                if (level_max > 32):
                    level_max = 32
                if game_name in ['reaction','numbers'] and level_max > 16:
                    level_max = 16
                
                level_arr = numpy.append(level_arr,level_max)
                
                
                
                # June 14
                if len(StimulusAppear_window_restricted) > 0:
                    # count the number of incorect answers
                    for ii in range(StimulusAppear_window_restricted.index.max(),StimulusAppear_window_restricted.index.max()+100000):
                        if ii in df['type'].index:
                            if df['type'].loc[ii] == 'payloadType_gameStateStart':
                               begining_of_new_game = ii
                               break
                    Stimulus_k = 0
                    Corr_k = 0
                    Incorr_k = 0
                    for k in range(StimulusAppear_window_restricted.index.min(), begining_of_new_game):
                        if k in df['type'].index:
                            if df['type'].loc[k] == 'payloadType_tiltStimulusAppear' and df['tiltTrialsName'].loc[k] == 'tiltTrialName_' + game_name:
                                Stimulus_k += 1
                            if df['type'].loc[k] == 'payloadType_tiltCorrect':
                                Corr_k += 1
                            if df['type'].loc[k] == 'payloadType_tiltIncorrect':
                                Incorr_k += 1
                    if Incorr_k > 6:
                        Incorr_k = 6
                    if Corr_k > Stimulus_k:
                        Corr_k = Stimulus_k
                else:
                    Stimulus_k, Incorr_k, Corr_k = 0,0,0
                # June 14 End
                Incorr_Total_array.append(Incorr_k)
                
                
            
                
                
                # report label for Correct/Incorrect
                for i in range(0,len(tiltCorrectIncorrect_window_restricted)):
                    current_label_text = tiltCorrectIncorrect_window_restricted.iloc[i,-1]
                    if current_label_text == 'payloadType_tiltCorrect':
                        corrIncorr_label = numpy.append(corrIncorr_label,1)
                        temporary_label.append(1)                
                    else:
                        corrIncorr_label = numpy.append(corrIncorr_label,0)
                        temporary_label.append(0)
                    # if the stimulus contains a star
                    if type(StimulusAppear_window_restricted['tiltTrialArr'].iloc[i]) == str:
                        if 'sr' in StimulusAppear_window_restricted['tiltTrialArr'].iloc[i]:
                            temporal_arr_st_1 = 1
                            temporal_arr_st.append(1)
                            array_st_all = numpy.append(array_st_all,1)
                        else:
                            temporal_arr_st.append(0)
                            array_st_all = numpy.append(array_st_all,0)
                            temporal_arr_st_1 = 0                    
                    else:
                        temporal_arr_st.append(0)
                        array_st_all = numpy.append(array_st_all,0)
                        temporal_arr_st_1 = 0
                    
                flips_arr = numpy.append(flips_arr, Flips)
                
                # June 12
                # if answer incorrect, check if the answer corresponds 
                # to conjunction visual lure
                for i_slot in range(0,len(Trial['tiltSlot'])):
                    String = Trial['tiltTrialArr'].iloc[i_slot]
                    if isinstance(Trial['tiltTrialArr'].iloc[i_slot], str):
                        ReLis = re.findall('(\w+)', String)
                        if temporary_label[i_slot] == 0:
                            for i, word in enumerate(ReLis):
                                TempWordx = re.findall('x$',word)
                                if TempWordx != []:
                                    # x_on is 0,1,2,3, corresponding to L,R,U,D
                                    x_on = i
                            if re.findall('([^x]+)',ReLis[x_on])[0][0] == 'Q':
                               lure = 'O'+re.findall('([^x]+)',ReLis[x_on])[0][1:]
                            if re.findall('([^x]+)',ReLis[x_on])[0][0] == 'O':
                               lure = 'Q'+re.findall('([^x]+)',ReLis[x_on])[0][1:]
                            # if lure present 
                            if lure in ReLis:
                                lure_index = ReLis.index(lure)
                            
                                index_Incorrect = tiltCorrectIncorrect_window_restricted.index[i_slot]
                                
                                for ii in range(index_Incorrect,Moves_window.index.max()):
                                    if ii in Moves_window.index:
                                       index_Moves_window = ii
                                       break
                                move_direction = re.findall('[^_]+$',Moves_window['tiltCursorTo'].loc[index_Moves_window])    
                                move_dir_index = ['left','right','up','down'].index(move_direction[0])
                                if move_dir_index == lure_index:
                                    # fallen for the lure
                                    fallen_for_lure = 1
                                    fallen_for_lure_temp.append(1)
                                else:
                                    fallen_for_lure = 0
                                    fallen_for_lure_temp.append(0)
                            else:
                                fallen_for_lure_temp.append(0)
                        else: 
                            fallen_for_lure_temp.append(0)
                    else:
                        fallen_for_lure_temp.append(0)
                
                fallen_for_lure_temp_all.append(fallen_for_lure_temp)
                
                
                        
                # June 14 Begin // Modification on 17 June
                fallen_for_opposite_temp = []
                # detecting if the star results in the error of opposite direction
                for i_slot in range(0,len(Trial['tiltSlot'])):
                    String = Trial['tiltTrialArr'].iloc[i_slot]
                    if isinstance(Trial['tiltTrialArr'].iloc[i_slot], str):
                        ReLis = re.findall('(\w+)', String)
                        if temporary_label[i_slot] == 0 and ('sr' in StimulusAppear_window_restricted['tiltTrialArr'].iloc[i_slot]):
                            for i, word in enumerate(ReLis):
                                TempWordx = re.findall('x$',word)
                                if TempWordx != []:
                                    # x_on is 0,1,2,3, corresponding to L,R,U,D
                                    x_on = i
                            
                            index_Incorrect = tiltCorrectIncorrect_window_restricted.index[i_slot]
                                
                            for ii in range(index_Incorrect,Moves_window.index.max()):
                                if ii in Moves_window.index:
                                    index_Moves_window = ii
                                    break
                            if index_Moves_window in Moves_window.index:
                                move_direction = re.findall('[^_]+$',Moves_window['tiltCursorTo'].loc[index_Moves_window])    
                                move_dir_index = ['left','right','up','down'].index(move_direction[0])
                                if (move_dir_index == 0 and x_on == 1) or\
                                    (move_dir_index == 1 and x_on == 0) or\
                                    (move_dir_index == 2 and x_on == 3) or\
                                    (move_dir_index == 3 and x_on == 2):
                                    # fallen for the opposite
                                    fallen_for_opposite = 1
                                    fallen_for_opposite_temp.append(1)
                                else:
                                    fallen_for_opposite = 0
                                    fallen_for_opposite_temp.append(0)
                            else:
                                fallen_for_opposite_temp.append(0)
                        else: 
                            fallen_for_opposite_temp.append(0)
                    else:
                        fallen_for_opposite_temp.append(0)
                
                fallen_for_opposite_temp_all.append(fallen_for_opposite_temp)
                
                
                # June 14 End
                
                        
                            
                window = (tiltCorrectIncorrect_window_restricted.values[:,0] - StimulusAppear_window_restricted.values[:,0]).astype(float)
                for i in range(0,len(window)):
                    # discard those windows longer than 15 seconds
                    if abs(window[i])>15:
                        window[i] = numpy.nan
                
                avg_window = numpy.nanmean(window)
                std_window = numpy.nanstd(window)
                avg = numpy.append(avg, avg_window)
                std = numpy.append(std, std_window)
                time_i = tiltCorrectIncorrect_window_restricted.values[0,0]
                time_ = numpy.append(time_, time_i)
                
                ## All values!
                time_event_all_values = numpy.append(time_event_all_values,window)
                
                reaction_times = []
                reaction_times_turning = []
                
                L1_trigger_all_temp = []
                L1_remaining_all_temp = []
                SE_y_all_temp = []
                SE_z_all_temp = []
                MD_trigger_all_temp = []
                MD_remaining_all_temp = []
                flips_trigger_all_temp = []
                flips_remaining_all_temp = []
                signal_trigger_all_temp = numpy.array([])
                signal_remaining_all_temp = numpy.array([])
                
                #if len(StimulusAppear_window) > 0:
                #        sys.exit("Check this window 1")
                #
                # Mark the direction of target
                target_ar = []
                for i_slot in range(0,len(Trial['tiltSlot'])):
                    String = Trial['tiltTrialArr'].iloc[i_slot]
                    if isinstance(Trial['tiltTrialArr'].iloc[i_slot], str):
                        ReLis = re.findall('(\w+)', String)
                        for i, word in enumerate(ReLis):
                            TempWordx = re.findall('x$',word)
                            if TempWordx != []:
                                # x_on is 0,1,2,3, corresponding to L,R,U,D
                                x_on = i
                                target_ar.append(x_on)
                            #else:
                            #    target_ar.append(float('nan'))
                    else:
                        target_ar.append(float('nan'))

                assert len(target_ar) == len(tiltCorrectIncorrect_window_restricted), 'Mismatch'
                
                for sub_window_i in range(0, len(tiltCorrectIncorrect_window_restricted)):
                    ###
                    ###
                    ### Here goes the code for quaternions.
                    ###
                    ###
                    considered_data = considered_data + 1
                    Stim_Appear = StimulusAppear_window_restricted.values[sub_window_i,1]
                    Tilt_Outcome = tiltCorrectIncorrect_window_restricted.values[sub_window_i,1]
                    
                    quat_raw_sub_window = quat_raw[(quat_raw.iloc[:,0]>= Stim_Appear) & (quat_raw.iloc[:,0]<=Tilt_Outcome)]
                    
                    #quat_raw_sub_window.sort_values(by='motionAt')
                    t = quat_raw_sub_window.iloc[:,0]
                    #if len(quat_raw_sub_window) > 0:
                    #    sys.exit("Check this window")
                    
                    #if not(all(numpy.diff(t.values)>0)):
                    #    sys.exit("time scrambled")
                    
                    time_session = StimulusAppear_window_restricted['at'].iloc[0]
                    time_session_all.append(time_session)    
                    
                    if sub_window_i < len(tiltCorrectIncorrect_window_restricted)-1:
                        for i in range(StimulusAppear_window_restricted.index[sub_window_i],StimulusAppear_window_restricted.index[sub_window_i+1]):
                            if i in df.index:
                                if (df.loc[i]['tiltCentreAtt_x']!= 0.0 and not(pandas.isnull(df.loc[i]['tiltCentreAtt_x']))):
                                    # get reference quaternion
                                    ref_quaternion = df.loc[i][['tiltCentreAtt_x', 'tiltCentreAtt_y', 'tiltCentreAtt_z', 'tiltCentreAtt_w']].values
                                    ref_quaternion[3] = numpy.sqrt(1-numpy.sum(ref_quaternion[0:3]**2))
                    #print(ref_quaternion)
                    angles = quaternion_to_euler(quaternion_multiply(quat_raw_sub_window.values[:,1:],inverse_quaternion(ref_quaternion)))
                    
                    #temporal_arr_st_1 == 1 and
                    if  (all(numpy.diff(t.values)>0)) and angles.size != 0 and len(angles) > 15 and (t.values[-1]-t.values[0]) < 10 and (abs(angles[-1,:]).max() < 0.6):
                        #if (abs(angles[-1,:]).max() -   
                        index_of_triggering = abs(angles[-1,0:2]).argmax()                
                        triggering_angle = angles[:,index_of_triggering]
                        
                        
                        remaining_two_indexes = list(set([0,1]) - set([index_of_triggering]))
        
                    # reaction time in ms
                    #reaction_time = float(t.values[tai]-t.values[0])
                    #reaction_times.append(reaction_time)
                        turning_point_index = CUMSUM_flip(triggering_angle,t.values)
                        #turning_point_index = onset_peaks(triggering_angle,t.values)
                        #turning_point_index = onset1(triggering_angle,t.values)
            
                        reaction_time_turning = float(t.values[turning_point_index]-t.values[0])
                        reaction_times_turning.append(reaction_time_turning)
                        
                        #"""
                        if temporary_label[sub_window_i] == 1:
                            # correct
                            sig_color = 'g'
                        else:
                            # incorrect
                            sig_color = 'r'
                            
                        if temporal_arr_st[sub_window_i] == 1:
                            # there is star
                            star_present = 'st'
                        else:
                            star_present = ' '
                        
                        if Flips[sub_window_i] == 1:
                            # there is a flip
                            flip_present = 'flip'
                        else:
                            flip_present = ' '
                        #"""
                        
                        plt.plot(t.values[turning_point_index], triggering_angle[turning_point_index], 'r.',label='RT')
                        plt.plot(t.values, triggering_angle,color=sig_color)
                        plt.text(t.values[turning_point_index],triggering_angle[turning_point_index]+0.01,star_present)
                        plt.text(t.values[turning_point_index],triggering_angle[turning_point_index]-0.05,flip_present)
                        plt.plot(t.values,angles[:,remaining_two_indexes],'b')
                        plt.grid(True)
                        plt.title('Triggering angles')
                        
                        #plt.plot(numpy.linspace(StimulusAppear_window_restricted.values[sub_window_i,0],StimulusAppear_window_restricted.values[sub_window_i,0]+float(t.values[turning_point_index]-t.values[0]), len(triggering_angle)),triggering_angle)
            
                        #plt.plot(StimulusAppear_window_restricted.values[sub_window_i,0]+reaction_time_turning, triggering_angle[turning_point_index], 'bo',label='Turning')
                        
                        print('ID: ' + str(ID)+"  Game: " + game_name +"   Session: " + str(window_i)+ '   Window: ' + str(sub_window_i) + '   Time: ' + str(datetime.fromtimestamp(StimulusAppear_window_restricted.values[sub_window_i,0])))
                        # print reference quaternion
                        # print(numpy.around(ref_quaternion.astype(float),3))
                        # make a break in the code
                        """
                        if temporary_label[sub_window_i] < 1:
                            suspicious_counter += 1
                            if suspicious_counter >6:
                                sys.exit("incorrect")
                        """
                        #if reaction_time_turning < 0.190:
                        #if temporal_arr_st[sub_window_i]==1 and temporary_label[sub_window_i]==10:
                        #    suspicious_counter += 1
                        #
                        # Normalization for integration
                        y = triggering_angle
                        z = angles[:,remaining_two_indexes]
                        if triggering_angle[-1] > 0:
                            y_norm = (y-y.min())/(y.max()-y.min())
                        else:
                            y_norm = 1-(y-y.min())/(y.max()-y.min())
                        z_norm = numpy.ravel((z-z.min())/(y.max()-y.min()))
                        #plt.plot(y_norm)
                        #plt.plot(z_norm)
                        # compute L1 norms
                        L1_trigger = integrate.simps(numpy.abs(1-y_norm), t.values)/(t.values[-1]-t.values[0])
                        L1_remaining = integrate.simps(numpy.abs(z_norm), t.values)/(t.values[-1]-t.values[0])
                        
                        
                        t_int = numpy.linspace(t.values[15], t.values[-1], 101)
                        #t_int = numpy.linspace(0, 100, 101)
                        y_int = numpy.interp(t_int, t.values[15:-1], angles[15:-1,0])
                        z_int = numpy.interp(t_int, t.values[15:-1], angles[15:-1,1])
                        
                        y_int_filtered = savgol_filter(y_int, 51, 3)
                        z_int_filtered = savgol_filter(z_int, 51, 3)
                        
                        time_integration = -1#turning_point_index#-1
                        dt = (t.values[time_integration]-t.values[0])
                        #dt = 1
                        # redefined L1 norms without normalization
                        if target_ar[sub_window_i] == 0:
                            L1_trigger = integrate.simps(numpy.abs(0.5-angles[0:time_integration,0]), t.values[0:time_integration])/dt#/(t.values[-1]-t.values[0])
                            L1_remaining = integrate.simps(numpy.abs(angles[0:time_integration,1]), t.values[0:time_integration])/dt#/(t.values[-1]-t.values[0])
                            MD_trigger = numpy.abs(numpy.min(angles[15:time_integration,0]))
                            MD_remaining = numpy.max(numpy.abs(angles[15:time_integration,1]))
                            flips_trigger = number_of_flips(y_int_filtered,t_int)
                            flips_remaining = number_of_flips(z_int_filtered,t_int)
                            signal_trigger = y_int
                            signal_remaining = z_int
                        elif target_ar[sub_window_i] == 1:
                            L1_trigger = integrate.simps(numpy.abs(-0.5-angles[0:time_integration,0]), t.values[0:time_integration])/dt#/(t.values[-1]-t.values[0])
                            L1_remaining = integrate.simps(numpy.abs(angles[0:time_integration,1]), t.values[0:time_integration])/dt#/(t.values[-1]-t.values[0])
                            MD_trigger = numpy.abs(numpy.max(angles[15:time_integration,0]))
                            MD_remaining = numpy.max(numpy.abs(angles[15:time_integration,1]))
                            flips_trigger = number_of_flips(y_int_filtered,t_int)
                            flips_remaining = number_of_flips(z_int_filtered,t_int)
                            signal_trigger = y_int
                            signal_remaining = z_int
                        elif target_ar[sub_window_i] == 2:
                            L1_trigger = integrate.simps(numpy.abs(-0.5-angles[0:time_integration,1]), t.values[0:time_integration])/dt#/(t.values[-1]-t.values[0])
                            L1_remaining = integrate.simps(numpy.abs(angles[0:time_integration,0]), t.values[0:time_integration])/dt#/(t.values[-1]-t.values[0])
                            MD_trigger = numpy.abs(numpy.max(angles[15:time_integration,1]))
                            MD_remaining = numpy.max(numpy.abs(angles[15:time_integration,0]))
                            flips_trigger = number_of_flips(z_int_filtered,t_int)
                            flips_remaining = number_of_flips(y_int_filtered,t_int)
                            signal_trigger = z_int
                            signal_remaining = y_int
                        elif target_ar[sub_window_i] == 3:
                            L1_trigger = integrate.simps(numpy.abs(0.5-angles[0:time_integration,1]), t.values[0:time_integration])/dt#/(t.values[-1]-t.values[0])
                            L1_remaining = integrate.simps(numpy.abs(angles[0:time_integration,0]), t.values[0:time_integration])/dt#/(t.values[-1]-t.values[0])
                            MD_trigger = numpy.abs(numpy.min(angles[15:time_integration,1]))
                            MD_remaining = numpy.max(numpy.abs(angles[15:time_integration,0]))
                            flips_trigger = number_of_flips(z_int_filtered,t_int)
                            flips_remaining = number_of_flips(y_int_filtered,t_int)
                            signal_trigger = z_int
                            signal_remaining = y_int
                        else:
                            L1_trigger = float('nan')
                            L1_remaining = float('nan')
                            MD_trigger = float('nan')
                            MD_remaining = float('nan')
                            flips_trigger = float('nan')
                            flips_remaining = float('nan')
                            signal_trigger = float('nan')*t_int
                            signal_remaining = float('nan')*t_int

                        
                        
                        #
                        # Sample entropy
                        r = 0.1
                        SE_y = SampEn(y_int, 5, r)
                        SE_z = SampEn(z_int, 5, 0.1)
        
                        
                        #
                        MD_trigger_all_temp.append(MD_trigger)
                        MD_remaining_all_temp.append(MD_remaining)
                        #
                        flips_trigger_all_temp.append(flips_trigger)
                        flips_remaining_all_temp.append(flips_remaining)
                        
                        L1_trigger_all_temp.append(L1_trigger)
                        L1_remaining_all_temp.append(L1_remaining)
                        SE_y_all_temp.append(SE_y)
                        SE_z_all_temp.append(SE_z)
                        
                        #
                        signal_trigger_all_temp = numpy.hstack((signal_trigger_all_temp,signal_trigger*numpy.sign(signal_trigger[-1])))
                        signal_remaining_all_temp = numpy.hstack((signal_remaining_all_temp,signal_remaining**numpy.sign(signal_trigger[-1])))

                        
                        
                        #if triggering_angle[-1] < 0:
                        #    sys.exit("report")
                        #    sys.exit('reaction time unrealistic')
        
                        
                    else:
                        reaction_times_turning.append(float('nan'))
                        L1_trigger_all_temp.append(float('nan'))
                        L1_remaining_all_temp.append(float('nan'))
                        SE_y_all_temp.append(float('nan'))
                        SE_z_all_temp.append(float('nan'))
                        MD_trigger_all_temp.append(float('nan'))
                        MD_remaining_all_temp.append(float('nan'))
                        flips_trigger_all_temp.append(float('nan'))
                        flips_remaining_all_temp.append(float('nan'))
                        signal_trigger_all_temp = numpy.hstack((signal_trigger_all_temp,float('nan')*t_int))
                        signal_remaining_all_temp = numpy.hstack((signal_remaining_all_temp,float('nan')*t_int))



                        
                    
                numpy.mean(reaction_times)    
                time_reaction_all = numpy.append(time_reaction_all,numpy.mean(reaction_times_turning))
                
                ## All values!
                time_reaction_all_values = numpy.append(time_reaction_all_values,reaction_times_turning)
                #
                L1_trigger_all = numpy.append(L1_trigger_all, L1_trigger_all_temp)
                L1_remaining_all = numpy.append(L1_remaining_all, L1_remaining_all_temp)
                SE_y_all = numpy.append(SE_y_all, SE_y_all_temp)
                SE_z_all = numpy.append(SE_z_all, SE_z_all_temp)
                #
                MD_trigger_all = numpy.append(MD_trigger_all,MD_trigger_all_temp)
                MD_remaining_all = numpy.append(MD_remaining_all,MD_remaining_all_temp)
                #
                flips_trigger_all = numpy.append(flips_trigger_all,flips_trigger_all_temp)
                flips_remaining_all = numpy.append(flips_remaining_all,flips_remaining_all_temp)
                #
                signal_trigger_all = numpy.hstack((signal_trigger_all, signal_trigger_all_temp))
                signal_remaining_all = numpy.hstack((signal_remaining_all, signal_remaining_all_temp))
                
                
                # Extract data for fitting GPs
                signals_trigger = numpy.zeros(shape=(int(len(signal_trigger_all)/101),101), dtype=float)
                for j in range(int(len(signal_trigger_all)/101)):
                    for i in range(101):
                        signals_trigger[j,i] = signal_trigger_all[j*101+i]
                        
                signals_remaining = numpy.zeros(shape=(int(len(signal_remaining_all)/101),101), dtype=float)
                for j in range(int(len(signals_remaining)/101)):
                    for i in range(101):
                        signals_remaining[j,i] = signal_remaining_all[j*101+i]

                                
                incorrect_window = df.loc[(df['type'] == 'payloadType_tiltIncorrect') & (df['ses'] == window_i) ][['at','atRaw','atFormatted']]
                index_restricted_to_game_name = [x for x in tiltCorrectIncorrect_window_restricted.index.tolist() if x in incorrect_window.index.tolist()]
                #incorrect_window_restricted = incorrect_window.ix[index_restricted_to_game_name]
                incorrect_window_restricted = incorrect_window.loc[index_restricted_to_game_name]
        
                n_incorrect = incorrect_window_restricted.shape[0]
                tot_incorrect.append(n_incorrect)
            if len(tiltCorrectIncorrect_window_restricted) > 0:
                avg_window_all = (tiltCorrectIncorrect_window_restricted.values[:,0] - StimulusAppear_window_restricted.values[:,0]).mean()/StimulusAppear_window_restricted.shape[0]
            else:
                avg_window_all = numpy.nan
            avg_all = numpy.append(avg_all,avg_window_all)
            
            # archive all the moves' durations and time of their occurrence
            #move_i = (tiltCorrectIncorrect_window_restricted.values[:,0] - StimulusAppear_window_restricted.values[:,0])
            #time_i = tiltCorrectIncorrect_window_restricted.values[:,0]
            
            #total_move = numpy.hstack((total_move,move_i))
            #total_time = numpy.hstack((total_time,time_i))
            #avg_all = numpy.append(avg_all,avg_window)
                
        print('Average time of compleation of ' + game_name + 'in consecutive sessions')
        print(avg)
        print('Number of incorrect answers')
        print(tot_incorrect)
        
        # for presentational purposes
        #t_delta = (scheduleStepBegan.iloc[-1,0] - scheduleStepBegan.iloc[0,0])*0.8
        #time_plot = 1.0*time_
        #for i in range(numpy.diff(time_plot).argmax(),len(time_plot)-1):
        #    time_plot[i+1] = time_plot[i+1]-(instance_3_start-baseline_end)*0.98
        
        
        r = (numpy.array([[datetime.fromtimestamp(time_[i]).day,datetime.fromtimestamp(time_[i]).hour] for i in range(0,len(time_))]))
        xlabel=[str(r[i][0])+'-'+str(r[i][1]) for i in range(0,len(r))]
        plt.figure(figsize=(12,6))
        plt.plot(time_,avg,'o')
        # Pad margins so that markers don't get clipped by the axes
        plt.margins(0.2)
        # Tweak spacing to prevent clipping of tick-labels
        plt.xticks(time_,xlabel, rotation='vertical')
        plt.xlabel('day - time')
        plt.ylabel('average time')
        plt.title('Performance on "' + game_name + '" game' )  
        plt.subplots_adjust(bottom=0.15)
        plt.grid(True)
        plt.tight_layout()
        plt.show() 
        
        
        print("Correlation between event and reaction times: " +
              str(numpy.corrcoef(time_event_all_values.astype(float),time_reaction_all_values.astype(float))[0,1]))
        import math
        def round_up(n, decimals=2):
            multiplier = 10 ** decimals
            return math.ceil(n * multiplier) / multiplier
        #plt.figure()
        #plt.hist(time_event_all_values.astype(float), 50, density=False, facecolor='g', alpha=0.75,label='event')
        #plt.hist(time_reaction_all_values.astype(float), 50, density=False, facecolor='r', alpha=0.75,label='reaction')
        #plt.title('Histograms of event registred and reaction times' + '  ( ' + game_name+ ')')
        #plt.xlabel('time [ms]')
        #plt.ylabel('Number')
        #plt.grid(True)
        #plt.legend()
        #style = dict(size=10, color='black')
        #plt.text(7.1, 65, "Event Mean = " + str(round_up(time_event_all_values.mean())), **style)
        #plt.text(7.1, 60, "Reaction Mean = " + str(round_up(time_reaction_all_values.mean())), **style)
        
        plt.show()
        
        # get rid of all nan's
        time_reaction_all_values_without_nans = numpy.array([time_reaction_all_values[i] for i in range(0,len(time_reaction_all_values)) if not(math.isnan(time_reaction_all_values[i]))])
        index_where_reaction_not_nans = numpy.array([i for i in range(0,len(time_reaction_all_values)) if not(math.isnan(time_reaction_all_values[i]))])
        index_where_reaction_nans = numpy.array([i for i in range(0,len(time_reaction_all_values)) if (math.isnan(time_reaction_all_values[i]))])
        time_event_all_values_redacted = numpy.array([i for i in range(0,len(time_event_all_values)) if not(math.isnan(time_reaction_all_values[i]))])
        time_event_all_values_where_reaction_not_nan = numpy.array([ time_event_all_values[i] for i in index_where_reaction_not_nans])
        
        L1_trigger_all_without_nans = numpy.array([ L1_trigger_all[i] for i in index_where_reaction_not_nans])
        L1_remaining_all_without_nans = numpy.array([ L1_remaining_all[i] for i in index_where_reaction_not_nans])
        SE_y_all_without_nans = numpy.array([ SE_y_all[i] for i in index_where_reaction_not_nans])
        SE_z_all_without_nan = numpy.array([ SE_z_all[i] for i in index_where_reaction_not_nans])
        
        print("Correlation between event and reaction times without nans: " +\
        str(numpy.corrcoef(time_reaction_all_values_without_nans,time_event_all_values_where_reaction_not_nan)[0][1]))
        
        print("Correlation between event times without nans and L1: " +\
        str(numpy.corrcoef(time_reaction_all_values_without_nans,L1_trigger_all_without_nans)[0][1]))
        
        
        if corrIncorr_label.shape[0] > 0:
            percent_of_incorrect_answers = len(numpy.where(corrIncorr_label==0)[0])/corrIncorr_label.shape[0]*100
        else:
            # no data
            percent_of_incorrect_answers = 0
        #CorrIncorVsStar = numpy.vstack([corrIncorr_label,array_st_all])
        
        
        #CorrIncorVsStar = pandas.DataFrame({'CorrIncorr':corrIncorr_label,'Star':array_st_all, 'Flip': flips_arr})
        
        fallen_for_opposite_temp_all_flattened = []
        for x in fallen_for_opposite_temp_all:
            for y in x:
                fallen_for_opposite_temp_all_flattened.append(y)
                
        fallen_for_lure_temp_all_flattened = []
        for x in fallen_for_lure_temp_all:
            for y in x:
                fallen_for_lure_temp_all_flattened.append(y)
        
        level_arr_with_nans = numpy.empty(len(time_event_all_values))
        level_arr_with_nans = numpy.full([len(time_event_all_values),0], numpy.nan)
        
        for i in range(0,len(level_arr)):
            level_arr_with_nans[i] = level_arr[i]
        for j in range(len(level_arr), len(time_event_all_values)):
            level_arr_with_nans[j] = numpy.nan
        
        out_1 = pandas.DataFrame({'Time':time_session_all,
                                'EventTime':time_event_all_values,
                                'RT':time_reaction_all_values,
                                'CorrIncorr':corrIncorr_label,
                                'Star':array_st_all, 
                                'Flip': flips_arr, 
                                'FallenForStar': fallen_for_opposite_temp_all_flattened,
                                'FallenForLure': fallen_for_lure_temp_all_flattened})
        
        out_2 = pandas.DataFrame({'LevelAchived': level_arr,
                                  'NumberOfFirstIncorrect': tot_incorrect,
                                  'NumberOfIncorrect':Incorr_Total_array})
        
        output = pandas.concat([out_1,out_2], axis=1)
        
        
        # ----------------------
        # ET's for specific condition
        if output.shape[0] > 0:
            ET_Correct = output.loc[(output['CorrIncorr'] == 1)]['EventTime'].mean()
            ET_Incorrect = output.loc[(output['CorrIncorr'] == 0)]['EventTime'].mean()
            ET_Flip = output.loc[(output['Flip'] == 1)]['EventTime'].mean()
            ET_NoFlip = output.loc[(output['Flip'] == 0)]['EventTime'].mean()

            ET_StarPresent = output.loc[(output['Star'] == 1)]['EventTime'].mean()
            ET_NoStarPresent = output.loc[(output['Star'] == 0)]['EventTime'].mean()

            ET_StarPresentIncorrect = output.loc[(output['Star'] == 1)& (output['CorrIncorr'] == 0)]['EventTime'].mean()
            ET_StarPresentCorrect = output.loc[(output['Star'] == 1)& (output['CorrIncorr'] == 1)]['EventTime'].mean()
            ET_StarFlip = output.loc[(output['Star'] == 1)& (output['Flip'] == 1)]['EventTime'].mean()
            ET_StarAndFallenForLure = output.loc[(output['Star'] == 1)& (output['FallenForLure'] == 1)]['EventTime'].mean()
            ET_StarNoFlip = output.loc[(output['Star'] == 1)& (output['Flip'] == 1)]['EventTime'].mean()

        else:
            ET_Correct = numpy.nan
            ET_Incorrect = numpy.nan
            ET_Flip = numpy.nan
            ET_NoFlip = numpy.nan

            ET_StarPresent = numpy.nan
            ET_NoStarPresent = numpy.nan

            ET_StarPresentIncorrect = numpy.nan
            ET_StarPresentCorrect = numpy.nan
            ET_StarFlip = numpy.nan
            ET_StarAndFallenForLure = numpy.nan
            ET_StarNoFlip = numpy.nan
        
        # ----------------------
        if output.shape[0] > 0:
            Incorr_percent_with_Flip_and_Star = numpy.float64(output.loc[(output['CorrIncorr'] == 0) & (output['Star'] == 1) & (output['Flip'] == 1)].shape[0])/output.loc[(output['CorrIncorr'] == 0)].shape[0] * 100
            Corr_percent_with_Flip_and_Star = numpy.float64(output.loc[(output['CorrIncorr'] == 1) & (output['Star'] == 1) & (output['Flip'] == 1)].shape[0])/output.loc[(output['CorrIncorr'] == 1)].shape[0] * 100
        else:
            Incorr_percent_with_Flip_and_Star = numpy.nan
            Corr_percent_with_Flip_and_Star = numpy.nan
        # ----------------------
        
        if output.shape[0] > 0:
            TitalIncorr_percent = numpy.float64(output.loc[(output['CorrIncorr'] == 0)].shape[0])/output.shape[0] * 100
        else:
            TitalIncorr_percent = 0
        
        if output.shape[0] > 0:
            IncorrStar_percent = numpy.float64(output.loc[(output['CorrIncorr'] == 0) &  (output['Star'] == 1)].shape[0])/output.loc[(output['CorrIncorr'] == 0)].shape[0] * 100
        else:
            IncorrStar_percent = 0
        
        if output.shape[0] > 0:
            IncorrFollenForStar_percent = numpy.float64(output.loc[(output['CorrIncorr'] == 0) &  (output['Star'] == 1) &  (output['FallenForStar'] == 1)].shape[0])/output.loc[(output['CorrIncorr'] == 0)].shape[0] * 100
        else:
            IncorrFollenForStar_percent = 0
            
        if output.shape[0] > 0:
            IncorrWithoutStar_percent = numpy.float64(output.loc[(output['CorrIncorr'] == 0) &  (output['Star'] == 0)].shape[0])/output.loc[(output['CorrIncorr'] == 0)].shape[0] * 100
        else:
            IncorrWithoutStar_percent = 0
            
        if output.shape[0] > 0:
            IncorrFlip_percent = numpy.float64(output.loc[(output['CorrIncorr'] == 0) &  (output['Flip'] == 1)].shape[0])/output.loc[(output['CorrIncorr'] == 0)].shape[0] * 100
        else:
            IncorrFlip_percent = 0
            
        if output.shape[0] > 0:
            IncorrWithoutFlip_percent = numpy.float64(output.loc[(output['CorrIncorr'] == 0) &  (output['Flip'] == 0)].shape[0])/output.loc[(output['CorrIncorr'] == 0)].shape[0] * 100
        else:
            IncorrWithoutFlip_percent = 0
            
        if output.shape[0] > 0:
            IncorrWithLure = numpy.float64(output.loc[(output['CorrIncorr'] == 0) &  (output['FallenForLure'] == 1)].shape[0])/output.loc[(output['CorrIncorr'] == 0)].shape[0] * 100
        else:
            IncorrWithLure = 0
        
        if output.shape[0] > 0:
            second_third_per = numpy.float64(sum(numpy.array(Incorr_Total_array)-numpy.array(tot_incorrect)))/len(output.loc[(output['CorrIncorr'] == 0)])*100
        else:
            second_third_per = 0
        
        """    
        if CorrIncorVsStar.shape[0] > 0:
            IncorrStar_percent = CorrIncorVsStar.loc[(CorrIncorVsStar['CorrIncorr'] == 0) &  (CorrIncorVsStar['Star'] == 1)].shape[0]/CorrIncorVsStar.shape[0] * 100
        else:
            IncorrStar_percent = 0
            
        if CorrIncorVsStar.shape[0] > 0:
            IncorrWithoutStar_percent = CorrIncorVsStar.loc[(CorrIncorVsStar['CorrIncorr'] == 0) &  (CorrIncorVsStar['Star'] == 0)].shape[0]/CorrIncorVsStar.shape[0] * 100
        else:
            IncorrWithoutStar_percent = 0
            
        if CorrIncorVsStar.shape[0] > 0:
            IncorrFlip_percent = CorrIncorVsStar.loc[(CorrIncorVsStar['CorrIncorr'] == 0) &  (CorrIncorVsStar['Flip'] == 1)].shape[0]/CorrIncorVsStar.shape[0] * 100
        else:
            IncorrFlip_percent = 0
            
        if CorrIncorVsStar.shape[0] > 0:
            IncorrWithoutFlip_percent = CorrIncorVsStar.loc[(CorrIncorVsStar['CorrIncorr'] == 0) &  (CorrIncorVsStar['Flip'] == 0)].shape[0]/CorrIncorVsStar.shape[0] * 100
        else:
            IncorrWithoutFlip_percent = 0
        """
            
        print('Total incorrect (%):          ',TitalIncorr_percent)
        
        print('Incorrect with Star (%):      ',IncorrStar_percent)
        print('Fallen for Star (%):          ',IncorrFollenForStar_percent)
        print('Incorrect without Star (%):   ',IncorrWithoutStar_percent)
        print('Incorrect with flip (%):      ',IncorrFlip_percent)
        print('Incorrect without flip (%):   ',IncorrWithoutFlip_percent)
        
        #print('Total incorrect (%):          ',TitalIncorr_percent)
        print('Total incorrect via flip (%): ',IncorrFlip_percent+IncorrWithoutFlip_percent)
        print('Total incorr. with lure (%):  ',IncorrWithLure)
        print('Incorr. 2nd and 3rd attempt:  ',second_third_per)
        
        plt.figure()
        plt.hist(time_event_all_values_where_reaction_not_nan.astype(float), 50, density=False, facecolor='g', alpha=0.75,label='event')
        plt.hist(time_reaction_all_values_without_nans.astype(float), 50, density=False, facecolor='r', alpha=0.75,label='reaction')
        plt.title('Histograms of event registred and reaction times' + '  (' + game_name+ ')')
        plt.xlabel('time [s]')
        plt.ylabel('Number of tasks')
        plt.grid(True)
        plt.legend()
        style = dict(size=10, color='black')
        #plt.text(7.1, 65, "Event Mean = " + str(round_up(time_event_all_values.mean())), **style)
        #plt.text(7.1, 60, "Reaction Mean = " + str(round_up(time_reaction_all_values.mean())), **style)
        
        plt.show()
        
        corrIncorr_label_withou_nans = numpy.array([corrIncorr_label[i] for i in index_where_reaction_not_nans])
        array_st_all_without_nans = numpy.array([array_st_all[i] for i in index_where_reaction_not_nans])
        
        tot_inocorr_ans = (IncorrStar_percent+IncorrWithoutStar_percent)
        #"""
        output_df = pandas.DataFrame([{'RT_'+game_name:time_reaction_all_values_without_nans.mean(),
                                      'RT-Std_'+game_name:time_reaction_all_values_without_nans.std(),
                                      'ET_'+game_name:time_event_all_values_where_reaction_not_nan.mean(),
                                      'ET-Std_'+game_name:time_event_all_values_where_reaction_not_nan.std(),
                                      'ET_Correct_'+game_name: ET_Correct,
                                      'ET_Incorrect_'+game_name: ET_Incorrect,
                                      'ET_Flip_'+game_name: ET_Flip,
                                      'ET_NoFlip_'+game_name:ET_NoFlip,
                                      'ET_StarPresent_'+game_name:ET_StarPresent,            
                                      'ET_NoStarPresent_'+game_name:ET_NoStarPresent,
                                      'ET_StarPresentIncorrect_'+game_name:ET_StarPresentIncorrect, 
                                      'ET_StarPresentCorrect_'+game_name:ET_StarPresentCorrect,
                                      'ET_StarFlip_'+game_name:ET_StarFlip,
                                      'ET_StarAndFallenForLure_'+game_name:ET_StarAndFallenForLure,
                                      'ET_StarNoFlip_'+game_name:ET_StarNoFlip,
                                      'IncorrectWithStar_'+game_name:IncorrStar_percent,
                                      'FallenForStar_'+game_name:IncorrFollenForStar_percent,
                                      'IncorrectWithoutStar_'+game_name:IncorrWithoutStar_percent,
                                      'IncorrectWithFlip_'+game_name:IncorrFlip_percent,
                                      'IncorrectWithoutFlip_'+game_name:IncorrWithoutFlip_percent,
                                      'IncorrectWithLure_'+game_name:IncorrWithLure,
                                      'IncorrectWithFlipAndStar': Incorr_percent_with_Flip_and_Star,
                                      'TotalIncorrect_'+game_name:TitalIncorr_percent,
                                      #'Incorrect2nd3rdAttempt_'+game_name:second_third_per,
                                      'LevelAchived_'+game_name:str(level_arr)}])
        #"""
        #output_df.loc[(output_df['CorrIncorr_'+game_name] == 1) & (output_df['Star_'+game_name] == 1)]['ReactionTime_'+game_name]
        waves = str(numpy.unique(df['instance'].values[~numpy.isnan(df['instance'].values)]).astype(int))
        
        output_all = pandas.concat([output_all, output_df], axis=1)
        #output_all.join(output_df)#, ignore_index = True)
        plt.close("all")
        
    plt.close("all")
    
    total_data_from_df = df.loc[(df['type'] == 'payloadType_tiltStimulusAppear') & (df['tiltTrialsName']!='tiltTrialName_unspecified')].shape[0]
    output_all.insert(0,'ID', ID)
    output_all.insert(1,'Wave', waves)
    output_all.insert(2,'Total_Windows', total_data_from_df)
    output_all.insert(3,'Total_Windows_Considered', considered_data)
    output_all.insert(4,'Sessions', ses.max()+1)
    
    output_all.to_csv(r'ForClair/out_file'+str(ID)+'.csv',index=False)    
        
    #print('Wave: ',numpy.unique(df['instance'].values[~numpy.isnan(df['instance'].values)]).astype(int))

"""
    outcome_measures = pandas.DataFrame({'ID':ID,\
    'reaction_Event_AVG':reaction_Event_AVG,\
    'reaction_Event_STD':reaction_Event_STD,\
    'reaction_RT_AVG':reaction_RT_AVG,\
    'reaction_RT_STD':reaction_RT_STD,\
    'numbers_Event_AVG':numbers_Event_AVG,\
    'numbers_Event_STD':numbers_Event_STD,\
    'numbers_RT_AVG':numbers_RT_AVG,\
    'numbers_RT_STD':numbers_RT_STD,\
    'numbersStarred_Event_AVG':numbersStarred_Event_AVG,\
    'numbersStarred_Event_STD':numbersStarred_Event_STD,\
    'numbersStarred_RT_AVG':numbersStarred_RT_AVG,\
    'numbersStarred_RT_STD':numbersStarred_RT_STD,\
    'doubleTrouble_Event_AVG':doubleTrouble_Event_AVG,\
    'doubleTroubled_Event_STD':doubleTroubled_Event_STD,\
    'doubleTrouble_RT_AVG':doubleTrouble_RT_AVG,\
    'doubleTrouble_RT_STD':doubleTrouble_RT_STD,\
    'troubleStarred_Event_AVG':troubleStarred_Event_AVG,\
    'troubleStarred_Event_STD':troubleStarred_Event_STD,\
    'troubleStarred_RT_AVG':troubleStarred_RT_AVG,\
    'troubleStarred_RT_STD':troubleStarred_RT_STD,\
    'reaction_L1_AVG':reaction_L1_AVG,\
    'reaction_L2_AVG':reaction_L2_AVG,\
    'reaction_L1_STD':reaction_L1_STD,\
    'reaction_L2_STD':reaction_L2_STD,\
    'numbers_L1_AVG':numbers_L1_AVG,\
    'numbers_L2_AVG':numbers_L2_AVG,\
    'numbers_L1_STD':numbers_L1_STD,\
    'numbers_L2_STD':numbers_L2_STD,\
    'numbersStarred_L1_AVG':numbersStarred_L1_AVG,\
    'numbersStarred_L2_AVG':numbersStarred_L2_AVG,\
    'numbersStarred_L1_STD':numbersStarred_L1_STD,\
    'numbersStarred_L2_STD':numbersStarred_L2_STD,\
    'doubleTrouble_L1_AVG':doubleTrouble_L1_AVG,\
    'doubleTrouble_L2_AVG':doubleTrouble_L2_AVG,\
    'doubleTrouble_L1_STD':doubleTrouble_L1_STD,\
    'doubleTrouble_L2_STD':doubleTrouble_L2_STD,\
    'troubleStarred_L1_AVG':troubleStarred_L1_AVG,\
    'troubleStarred_L2_AVG':troubleStarred_L2_AVG,\
    'troubleStarred_L1_STD':troubleStarred_L1_STD,\
    'troubleStarred_L2_STD':troubleStarred_L2_STD,\
    'reaction_SE1_AVG':reaction_SE1_AVG,\
    'reaction_SE2_AVG':reaction_SE2_AVG,\
    'reaction_SE1_STD':reaction_SE1_STD,\
    'reaction_SE2_STD':reaction_SE2_STD,\
    'numbers_SE1_AVG':numbers_SE1_AVG,\
    'numbers_SE2_AVG':numbers_SE2_AVG,\
    'numbers_SE1_STD':numbers_SE1_STD,\
    'numbers_SE2_STD':numbers_SE2_STD,\
    'numbersStarred_SE1_AVG':numbersStarred_SE1_AVG,\
    'numbersStarred_SE2_AVG':numbersStarred_SE2_AVG,\
    'numbersStarred_SE1_STD':numbersStarred_SE1_STD,\
    'numbersStarred_SE2_STD':numbersStarred_SE2_STD,\
    'doubleTrouble_SE1_AVG':doubleTrouble_SE1_AVG,\
    'doubleTrouble_SE2_AVG':doubleTrouble_SE2_AVG,\
    'doubleTrouble_SE1_STD':doubleTrouble_SE1_STD,\
    'doubleTrouble_SE2_STD':doubleTrouble_SE2_STD,\
    'troubleStarred_SE1_AVG':troubleStarred_SE1_AVG,\
    'troubleStarred_SE2_AVG':troubleStarred_SE2_AVG,\
    'troubleStarred_SE1_STD':troubleStarred_SE1_STD,\
    'troubleStarred_SE2_STD':troubleStarred_SE2_STD,\
    'numbers_incorrect':numbers_incorrect,\
    'numbersStarred_incorrect':numbersStarred_incorrect,\
    'doubleTrouble_incorrect':doubleTrouble_incorrect,\
    'troubleStarred_incorrect':troubleStarred_incorrect,\
    'numbersStarred_Sr_incorrect':numbersStarred_Sr_incorrect,\
    'troubleStarred_Sr_incorrect':troubleStarred_Sr_incorrect,\
    'numbersStarred_RT_with_Sr':numbersStarred_RT_with_Sr,\
    'numbersStarred_RT_without_Sr':numbersStarred_RT_without_Sr,\
    'troubleStarred_RT_with_Sr':troubleStarred_RT_with_Sr,\
    'troubleStarred_RT_without_Sr':troubleStarred_RT_without_Sr,\
    'Number_of_Sessions':numpy.int(df['ses'].max())},index=[0])
        
            
    
        
    outcome_measures.to_csv(r'out_file'+str(ID)+'.csv',index=False)
"""