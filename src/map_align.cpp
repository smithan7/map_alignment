//============================================================================
// Name        : map_align.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string>

#include "SatImg.h"

#include <vector>

using namespace std;
using namespace cv;

int main() {

	int map_num = 4;
	double lat1, lon1, lat0, lon0;
	double start_lat, start_lon, goal_lat, goal_lon;

	if( map_num == 0){
		lat1 = 44.538552;
		lon1 = -123.247446; // bottom right corner
		lat0 =  44.539847;
		lon0 = -123.251004; // top left corner

		start_lat = 44.539201;
		start_lon = -123.250343;
		goal_lat = 44.539300;
		goal_lon = -123.248151;
	}
	//harware 1
	else if( map_num == 1){
		lat1 = 44.537679;
		lon1 = -123.248297; // bottom right corner
		lat0 = 44.539295;
		lon0 = -123.249711; // top left corner

		start_lat = 44.537865;
		start_lon = -123.249370;
		goal_lat = 44.539070;
		goal_lon = -123.248856;
	}
	//harware 2
	else if( map_num == 2){
		lat1 = 44.537470;
		lon1 = -123.249107; // bottom right corner
		lat0 = 44.539048;
		lon0 = -123.250807; // top left corner

		start_lat = 44.537523;
		start_lon = -123.249329;
		goal_lat = 44.538916;
		goal_lon = -123.250282;
	}
	// test environment on OSU
	else if( map_num == 3){
		lat1 = 44.564965;
		lon1 = -123.270456; // bottom right corner
		lat0 = 44.565683;
		lon0 =  -123.272974; // top left corner

		start_lat = 44.564965;
		start_lon = -123.270456;
		goal_lat = 44.565683;
		goal_lon = -123.272974;
	}

	else if( map_num == 4){
		lat1 = 44.538162;
		lon1 = -123.247776; // bottom right corner
		lat0 = 44.539457;
		lon0 =  -123.250866; // top left corner

		start_lat = 44.539201;
		start_lon = -123.250343;
		goal_lat = 44.539300;
		goal_lon = -123.248151;
	}



	vector<double> start, goal;
	start.push_back( start_lon );
	start.push_back( start_lat );

	goal.push_back( goal_lon );
	goal.push_back( goal_lat );

	vector<double> corners;
	corners.push_back( lon0 );
	corners.push_back( lat0 );
	corners.push_back( lon1 );
	corners.push_back( lat1 );

	//SatImg satImg("/home/rdml/git/map_align/short_hardware/", "easy"+s, corners, start, goal);
    //SatImg satImg("/home/rdml/git/map_align/short_hardware/", "easy"+s, corners, start, goal);
	SatImg satImg("/home/andy/catkin_ws/src/map_align/data", "hardware4", corners, start, goal);

	cv::waitKey(0);

	return 0;
}
