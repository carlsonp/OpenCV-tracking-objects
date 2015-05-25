#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <cxcore.h>
#include <cvaux.h>

#include <time.h>
#include <math.h>
#include <vector>


/*
Patrick Carlson
HCI 575 - Computational Perception
3/4/10
Homework 3 - Task 2.d, Task 3, 4, 5
		Track all worms in videos
*/ 

using namespace std;

// various tracking parameters (in seconds)
const double MHI_DURATION = 1;
const double MAX_TIME_DELTA = 0.5;
const double MIN_TIME_DELTA = 0.05;
// number of cyclic frame buffer used for motion detection
// (should, probably, depend on FPS)
const int N = 2;

// ring image buffer
IplImage **buf = 0;
int last = 0;
int frame_counter = 0;

// temporary images
IplImage *mhi = 0; // MHI
IplImage *orient = 0; // orientation
IplImage *mask = 0; // valid orientation mask
IplImage *segmask = 0; // motion segmentation map
CvMemStorage* storage = 0; // temporary storage

// points used for creating path of worm
vector<CvPoint> past_points;
vector<int> last_update;

vector<CvPoint> to_process_points;

struct line_segment{
int x1;
int y1;
int x2;
int y2;
};
vector<line_segment> line_segments;

vector<int> line_segments_age;
const int MAX_LINE_AGE = 30; //amount of time the path line data is drawn before being discarded as old


double calcDistance(int x1, int y1, int x2, int y2)
{
	double distance;
	distance = sqrt(pow((x2-x1), 2) + pow((y2-y1), 2));
	return distance;
}

//max_distance is the maximum distance (in pixels) away a sequence can be from another worm before it is considered a separate worm
//last_update_die is the maximum amount of time in frames a worm can go without moving before it is removed
void  update_mhi( IplImage* img, IplImage* dst, int diff_threshold, int max_distance, int last_update_die )
{
    double timestamp = (double)clock()/CLOCKS_PER_SEC; // get current time in seconds
    CvSize size = cvSize(img->width,img->height); // get current frame size
    int i, idx1 = last, idx2;
    IplImage* silh;
    CvSeq* seq;
    CvRect comp_rect;

    // allocate images at the beginning or
    // reallocate them if the frame size is changed
    if( !mhi || mhi->width != size.width || mhi->height != size.height ) {
        if( buf == 0 ) {
            buf = (IplImage**)malloc(N*sizeof(buf[0]));
            memset( buf, 0, N*sizeof(buf[0]));
        }
        
        for( i = 0; i < N; i++ ) {
            cvReleaseImage( &buf[i] );
            buf[i] = cvCreateImage( size, IPL_DEPTH_8U, 1 );
            cvZero( buf[i] );
        }
        cvReleaseImage( &mhi );
        cvReleaseImage( &orient );
        cvReleaseImage( &segmask );
        cvReleaseImage( &mask );
        
        mhi = cvCreateImage( size, IPL_DEPTH_32F, 1 );
        cvZero( mhi ); // clear MHI at the beginning
        orient = cvCreateImage( size, IPL_DEPTH_32F, 1 );
        segmask = cvCreateImage( size, IPL_DEPTH_32F, 1 );
        mask = cvCreateImage( size, IPL_DEPTH_8U, 1 );
    }

    cvCvtColor( img, buf[last], CV_BGR2GRAY ); // convert frame to grayscale

    idx2 = (last + 1) % N; // index of (last - (N-1))th frame
    last = idx2;
	frame_counter++;

    silh = buf[idx2];
    cvAbsDiff( buf[idx1], buf[idx2], silh ); // get difference between frames
    
    cvThreshold( silh, silh, diff_threshold, 1, CV_THRESH_BINARY ); // and threshold it
	//cvSmooth(silh, silh, CV_MEDIAN, 3, 3); //run median smoothing to fill in tiny spots
    cvUpdateMotionHistory( silh, mhi, timestamp, MHI_DURATION ); // update MHI

    // convert MHI to blue 8u image
    cvCvtScale( mhi, mask, 255./MHI_DURATION, (MHI_DURATION - timestamp)*255./MHI_DURATION );
    cvZero( dst );
    cvCvtPlaneToPix( mask, 0, 0, 0, dst );

    // calculate motion gradient orientation and valid orientation mask
    cvCalcMotionGradient( mhi, mask, orient, MAX_TIME_DELTA, MIN_TIME_DELTA, 3 );
    
    if( !storage )
        storage = cvCreateMemStorage(0);
    else
        cvClearMemStorage(storage);
    
    // segment motion: get sequence of motion components
    // segmask is marked motion components map. It is not used further
    seq = cvSegmentMotion( mhi, segmask, storage, timestamp, MAX_TIME_DELTA );

	//printf("number sequences: %i\n", seq->total);
    // iterate through the motion components,
    // One more iteration (i == -1) corresponds to the whole image (global motion)
    for( i = -1; i < seq->total; i++ )
	{
		if( i < 0 ) { // case of the whole image
            comp_rect = cvRect( 0, 0, size.width, size.height ); //we don't really need this information
        }
        else { // i-th motion component
            comp_rect = ((CvConnectedComp*)cvGetSeqElem( seq, i ))->rect;
            if( comp_rect.width + comp_rect.height < 100 ) // reject very small components
                continue;

			// select component ROI
		    cvSetImageROI( silh, comp_rect );
		    cvSetImageROI( mhi, comp_rect );
		    cvSetImageROI( orient, comp_rect );
		    cvSetImageROI( mask, comp_rect );

			// calculate center of worm
			CvPoint temp_point = cvPoint( (comp_rect.x + comp_rect.width/2), (comp_rect.y + comp_rect.height/2) );
			//printf("Center: %i , %i \n", temp_point.x, temp_point.y);
			if (frame_counter > 6) //skip the first couple frames
			{
				//add point to the vector of points to be processed
				to_process_points.push_back(temp_point);
			}
			else
			{
				//just put all sequences as new worms since we can't really be sure at this point in time
				past_points.push_back(temp_point);
				last_update.push_back(frame_counter);
			}
			//printf("Frame counter %i \n", frame_counter);

		    cvResetImageROI( mhi );
		    cvResetImageROI( orient );
		    cvResetImageROI( mask );
		    cvResetImageROI( silh );
        }
    }
	//Finished going through sequences

	while (to_process_points.size() != 0)
	{
		//Now that we have all the possible movements points ready to go, they need to be matched to a worm
		//Iterate through all existing worms and find closest new movement point that is within the max_distance value
		int smallest_distance = 100000000;
		int new_index_value = -1;
		int old_index_value = -1;
		for (unsigned int i = 0; i < past_points.size(); i++)
		{
			for (unsigned int j = 0; j < to_process_points.size(); j++)
			{
				double dist = calcDistance(to_process_points[j].x, to_process_points[j].y, past_points[i].x, past_points[i].y);
				if (smallest_distance > dist && dist < max_distance && last_update[i] != frame_counter) //make sure we aren't adding more than one motion sequence to an existing worm
				{
					//Possible match found!
					smallest_distance = dist;
					new_index_value = j;
					old_index_value = i;
				}
			}
		}
		//We have the best match for a worm at this point
		if (smallest_distance == 100000000)
		{
			//At this point, whatever is left are considered new worms
			for (unsigned int i = 0; i < to_process_points.size(); i++)
			{
				past_points.push_back(to_process_points[i]);
				last_update.push_back(frame_counter);
				to_process_points.erase(to_process_points.begin()+i);
			}
		}
		else
		{
			//Found a match with an existing worm, draw the line
			line_segment ls;
			ls.x1 = past_points[old_index_value].x;
			ls.y1 = past_points[old_index_value].y;
			ls.x2 = to_process_points[new_index_value].x;
			ls.y2 = to_process_points[new_index_value].y;
			line_segments.push_back(ls);
			line_segments_age.push_back(frame_counter);

			//Replace old point with new point
			past_points[old_index_value] = to_process_points[new_index_value];

			//Update timestamp (frame) and new point
			last_update[old_index_value] = frame_counter;
			//Delete processed point
			to_process_points.erase(to_process_points.begin()+new_index_value);
		}
	}

	//Check for old worms that haven't moved, if found remove them
	for (unsigned int i = 0; i < past_points.size(); i++)
	{
		if (last_update_die < (frame_counter - last_update[i]))
		{
			//Remove old worm
			past_points.erase(past_points.begin()+i);
			last_update.erase(last_update.begin()+i);
		}
	}
}

//Modified from: http://stackoverflow.com/questions/1571683/opencv-image-on-image
void drawPath(IplImage* target, IplImage* source) {
    for (int x=0; x<source->width; x++) {
        for (int y=0; y<source->height; y++) {
            int r = cvGet2D(source, y, x).val[2];
            //int g = cvGet2D(source, y, x).val[1];
            //int b = cvGet2D(source, y, x).val[0];
			//printf("r,g,b %i,%i,%i \n", r,g,b);
            CvScalar bgr = cvScalar(0, 0, r);
			if (r != 0) //only draw on image if we find red
			{
            	cvSet2D(target, y, x, bgr);
			}
        }
    }
}

int main(int argc, char* argv[])
{
	//Working with video in OpenCV
	//http://www.cs.iit.edu/~agam/cs512/lect-notes/opencv-intro/opencv-intro.html#SECTION00070000000000000000

	//Code adapted from motempl.c example on course website

	//See pg. 341 of OpenCV book for more details

	CvCapture* capture = cvCreateFileCapture( argv[1] );

	CvVideoWriter *writer = 0;


	IplImage* motion = 0;
	IplImage* final = 0;
	IplImage* path = 0;

    if( capture )
    {
        cvNamedWindow( "Worm Tracking", 1 );
        
        for(;;)
        {
            IplImage* image;
            if( !cvGrabFrame( capture ))
                break;
            image = cvRetrieveFrame( capture );

            if( image )
            {
                if( !motion ) //initialize everything if we are on first run through
                {
					//setup video output
					int isColor = 1;
					int fps     = (int) cvGetCaptureProperty(capture, CV_CAP_PROP_FPS); //get FPS from original capture
					printf("Video FPS: %i\n", fps);
					int frameW  = image->width;
					int frameH  = image->height;
					/*
					Output Video Codecs:
					
					CV_FOURCC('P','I','M','1')    = MPEG-1 codec
					CV_FOURCC('M','J','P','G')    = motion-jpeg codec (does not work well)
					CV_FOURCC('M', 'P', '4', '2') = MPEG-4.2 codec
					CV_FOURCC('D', 'I', 'V', '3') = MPEG-4.3 codec
					CV_FOURCC('D', 'I', 'V', 'X') = MPEG-4 codec
					CV_FOURCC('U', '2', '6', '3') = H263 codec
					CV_FOURCC('I', '2', '6', '3') = H263I codec
					CV_FOURCC('F', 'L', 'V', '1') = FLV1 codec
					*/
					writer = cvCreateVideoWriter("output.avi",CV_FOURCC('D','I','V','X'),fps,cvSize(frameW,frameH),isColor);

                    motion = cvCreateImage( cvSize(image->width,image->height), IPL_DEPTH_8U, 3 );
					path = cvCreateImage( cvSize(image->width,image->height), IPL_DEPTH_8U, 3 );
					final = cvCreateImage( cvSize(image->width,image->height), IPL_DEPTH_8U, 3 );
                    cvZero( motion );
                    motion->origin = image->origin;
                }
            }

            update_mhi( image, motion, 10, 30, 15 );

			//Zero out old path
			cvZero(path);
			//Draw new path
			for (unsigned int i = 0; i < line_segments.size(); i++)
			{
				if (line_segments_age[i] > (frame_counter - MAX_LINE_AGE))
					cvLine(path, cvPoint(line_segments[i].x1, line_segments[i].y2), cvPoint(line_segments[i].x2, line_segments[i].y2), cvScalar(0,0,255), 1);
				else
				{
					line_segments.erase(line_segments.begin()+i);
					line_segments_age.erase(line_segments_age.begin()+i);
				}
			}

			cvCopy(motion, final);
			drawPath(final, path);
            cvShowImage( "Worm Tracking", final );
			cvWriteFrame(writer,final); // add the frame to the output video

            if( cvWaitKey(10) >= 0 )
                break;
        }
		cvReleaseImage( &motion );
		cvReleaseImage( &final );
		cvReleaseImage( &path );
        cvReleaseCapture( &capture );
		cvReleaseVideoWriter(&writer);
        cvDestroyWindow( "Worm Tracking" );
    }
	 
	return 0;
}
