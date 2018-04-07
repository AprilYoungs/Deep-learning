import time
import pandas as pd
import numpy as np

CITY_DATA = { 'chicago': 'chicago.csv',
              'new york city': 'new_york_city.csv',
              'washington': 'washington.csv' }
MONTH_DATA = ['january', 'february', 'march' ,'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']


def get_filters():
    """
    Asks user to specify a city, month, and day to analyze.

    Returns:
        (str) city - name of the city to analyze
        (str) month - name of the month to filter by, or "all" to apply no month filter
        (str) day - name of the day of week to filter by, or "all" to apply no day filter
    """
    print('Hello! Let\'s explore some US bikeshare data!')
    # get user input for city (chicago, new york city, washington). HINT: Use a while loop to handle invalid inputs

    while True:
        city = input('Which city do you want to explore?\n\
                      1.chicago\n\
                      2.new york city\n\
                      3.washington\n')
        city = city.strip()
        city = city.lower()
        cityList = ['chicago', 'new york city', 'washington']
        if city in ['1','2','3']:
            city = cityList[int(city)-1]
            break
        elif city in cityList:
            break

        else:
            print('-'*40)
            print('Error input,please choose one city above')
            print('-'*40)

    # get user input for month (all, january, february, ... , june)
    while True:
        month = input('Which month do you want to explore?\n\
                      0.all\n\
                      1.january\n\
                      2.february\n\
                      3.march\n\
                      4.april\n\
                      5.may\n\
                      6.june\n\
                      7.july\n\
                      8.august\n\
                      9.september\n\
                      10.october\n\
                      11.november\n\
                      12.december\n')
        month = month.strip()
        month = month.lower()
        monthList = ['all', 'january', 'february', 'march' ,'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
        tokens = [str(i) for i in range(13)]
        if month in tokens:
            month = monthList[int(month)]
            break
        elif month in monthList:
            break
        else:
            print('-'*40)
            print('Error input,please choose one above')
            print('-'*40)

    # get user input for day of week (all, monday, tuesday, ... sunday)
    while True:
        day = input('Which day of the week do you want to explore?\n\
                      0.all\n\
                      1.monday\n\
                      2.tuesday\n\
                      3.wednesday\n\
                      4.thursday\n\
                      5.friday\n\
                      6.saturday\n\
                      7.sunday\n')
        day = day.strip()
        day = day.lower()
        dayList = ['all', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        tokens = [str(i) for i in range(8)]
        if day in tokens:
            day = dayList[int(day)]
            break
        elif day in dayList:
            break
        else:
            print('-'*40)
            print('Error input,please choose one above')
            print('-'*40)

    print('-'*40)
    return city, month, day


def load_data(city, month, day):
    """
    Loads data for the specified city and filters by month and day if applicable.

    Args:
        (str) city - name of the city to analyze
        (str) month - name of the month to filter by, or "all" to apply no month filter
        (str) day - name of the day of week to filter by, or "all" to apply no day filter
    Returns:
        df - Pandas DataFrame containing city data filtered by month and day
    """
    path = CITY_DATA[city]
    df = pd.read_csv(path)
    df.dropna(inplace=True)

    df['date'] = pd.to_datetime(df['Start Time'])
    df['month'] = df['date'].apply(lambda x: MONTH_DATA[x.month-1])
    df['weekday'] = df['date'].apply(lambda x: x.weekday_name.lower())

    # filter month
    if month != 'all':
        df = df[df['month'] == month]

    #filter day
    if day != 'all':
        df = df[df['weekday'] == day]

    #drop the assist columns
    # df.drop(['date','month','weekday'], axis=1, inplace=True)

    return df


def time_stats(df):
    """Displays statistics on the most frequent times of travel."""

    print('\nCalculating The Most Frequent Times of Travel...\n')
    start_time = time.time()

    if len(df) > 0:
        # display the most common month
        common_month = df.month.value_counts().idxmax()
        print('The most common month is %s' % common_month)

        # display the most common day of week
        common_day = df.weekday.value_counts().idxmax()
        print('The most common day of week is %s' % common_day)

        # display the most common start hour
        df['start_hour'] = df.date.apply(lambda x: x.hour)
        common_start_hour = df.start_hour.value_counts().idxmax()
        print('The most common start hour is %s' % common_start_hour)
    else:
        print("These is no data with this filter")

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)


def station_stats(df):
    """Displays statistics on the most popular stations and trip."""

    print('\nCalculating The Most Popular Stations and Trip...\n')
    start_time = time.time()

    if len(df) > 0:
        # display most commonly used start station
        start_station = df['Start Station'].value_counts().idxmax()
        print('The most commonly used start station is %s' % start_station)

        # display most commonly used end station
        end_station = df['End Station'].value_counts().idxmax()
        print('The most commonly used end station is %s' % end_station)


        # display most frequent combination of start station and end station trip
        df['S_E_Station'] = df['Start Station'] + ' to ' + df['End Station']
        combination = df['S_E_Station'].value_counts().idxmax()
        print('The most frequent combination of start station and end station trip is %s' % combination)
    else:
        print("These is no data with this filter")

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)


def trip_duration_stats(df):
    """Displays statistics on the total and average trip duration."""

    print('\nCalculating Trip Duration...\n')
    start_time = time.time()

    # display total travel time
    total_sec = df['Trip Duration'].sum()
    if total_sec == 0:
        print("These is no data with this filter")
    else:
        print('The total travel time is %d secs,that\'s %.2f hours' % (total_sec, total_sec/3600))

        # display mean travel time
        mean_sec = df['Trip Duration'].mean()
        print('The mean travel time is %.3f secs,that\'s %.2f mins' % (mean_sec, mean_sec/60))


    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)


def user_stats(df):
    """Displays statistics on bikeshare users."""

    print('\nCalculating User Stats...\n')
    start_time = time.time()

    if len(df) > 0:
        # Display counts of user types
        user_types = df['User Type'].value_counts()
        for s in user_types.index:
            print('User type {} with a count of {}'.format(s, user_types.loc[s]))

        # Display counts of gender
        if 'Gender' in df.columns:
            gender_counts = df['Gender'].value_counts()
            for s in gender_counts.index:
                print('Count of {} is {}'.format(s, gender_counts.loc[s]))
        else:
            print("The data didn't has a 'Gender' column")

        # Display earliest, most recent, and most common year of birth
        if 'Birth Year' in df.columns:
            earliest = df['Birth Year'].min()
            recent = df['Birth Year'].max()
            common_year = df['Birth Year'].value_counts().idxmax()
            print('The earliest year of birth is {}, the recent year of birth is {}, the most common year of birth is {}'.format(earliest, recent, common_year))
        else:
            print("The data didn't has a 'Birth Year' column")
    else:
        print("These is no data with this filter")

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)

def display_data(df):
    '''
     Displays five lines of data if the user specifies that they would like to.
     After displaying five lines, ask the user if they would like to see five
      more, continuing asking until they say stop.
    '''
    start = 0

    print('Would you like to see the data?')
    while True:
        like = input('Input y for Yes,n for No:')
        like = like.strip().lower()
        if like == 'y':
            print(df.iloc[start: start + 5])
            start += 5
            while True:
                like = input('Wanna see more?\n\
                Input y for Yes,n for No:')
                like = like.strip().lower()
                if like == 'y':
                    print(df.iloc[start: start + 5])
                    start += 5
                elif like == 'n':
                    break
                else:
                    print("Error input,I didn't get that")
            break
        elif like == 'n':
            break
        else:
            print("Error input,I didn't get that")




def main():
    while True:
        city, month, day = get_filters()
        df = load_data(city, month, day)

        display_data(df)
        time_stats(df)
        station_stats(df)
        trip_duration_stats(df)
        user_stats(df)

        restart = input('\nWould you like to restart? Enter yes or no.\n')
        if restart.lower() != 'yes':
            break


if __name__ == "__main__":
	main()
