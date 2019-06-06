import numpy as np
import cv2
import os
import datetime
import matplotlib.pyplot as plt
import pdb
import sys


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    y = y.astype(int)    
    x = x.astype(int)    
    fx, fy = flow[y,x].T    
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    if len(img.shape) < 3:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else: 
        vis = img
    cv2.polylines(vis, lines, 0, (0, 255, 0))    
    for (x1, y1), (x2, y2) in lines:    
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)    
    return vis    

def intersect(x1,y1,dx1,dy1,x2,y2,dx2,dy2):
    t = (dy2*(x2-x1)-dx2*(y2-y1))/(dx1*dy2-dx2*dy1)
    return (x1+dx1*t,y1+dy1*t)

def flow_adjust(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx)
    v = np.sqrt(fx*fx+fy*fy)
    ang_median = np.median(ang)
    v_median = np.median(v)
    
    sample_x_pts = np.random.randint(h, size=(2, 10))
    sample_y_pts = np.random.randint(w,size=(2,10))
    intersection_pts=[]
    pts_angle = []
    for i in range(10):
        x1, y1 = sample_x_pts[0,i], sample_y_pts[0,i]
        x2, y2 = sample_x_pts[1,i], sample_y_pts[1,i]
         
        dx1, dy1 = flow[x1, y1]
        dx2, dy2 = flow[x2, y2]
        
        pts_angle.append(ang[x1, y1]* 180 / np.pi)
        pts_angle.append(ang[x2, y2]* 180 / np.pi)
        
        intersection_pts.append(intersect(x1,y1,dx1,dy1,x2,y2,dx2,dy2))
        
#     print('angle: ',pts_angle)
#     print("points: ", intersection_pts)
    inter_pts = np.asarray(intersection_pts)
    
#   print(np.nonzero(ang==np.median(ang)),np.nonzero(v==np.median(v)))
    
#     print(np.median(ang) ,np.median(v))
    
    # get median through angle and norm
    fx_median_1 = np.tan(np.median(ang))*np.median(v)
    fy_median_1 = np.tan(np.median(ang))*np.median(v)


#     print('median_vector: ', (fx_median_1,fy_median_1),'\n','median_flow_magnitude: ',np.linalg.norm([fx_median_1,fy_median_1]))
    
    # based on magnitude to adjust percentile of shiting vector
    if abs(np.median(fx)) > 2: 
        percentile_x = 50
    else:
        percentile_x = 70
        
    if abs(np.median(fy)) > 2:
        percentile_y = 50
    else:
        percentile_y = 70    
    
    if np.median(fx)>=0: 
        fx_adj = np.percentile(fx,percentile_x)
    else:  
        fx_adj = np.percentile(fx,100-percentile_x)
        
    if np.median(fy)>=0:
        fy_adj = np.percentile(fy,percentile_y)
    else:
        fy_adj = np.percentile(fy,100-percentile_y)
    
#     print((fx_median_1,fy_median_1), (fx_adj,fy_adj))
    
    quantile_x = (np.percentile(fx,25),np.percentile(fx,50),np.percentile(fx,75),np.percentile(fx,90))
    quantile_y = (np.percentile(fy,25),np.percentile(fy,50),np.percentile(fy,75),np.percentile(fy,90))
    quantile_ang = (np.percentile(ang,25),np.percentile(ang,50),np.percentile(ang,75),np.percentile(ang,90))
    
    return (fx_adj,fy_adj, quantile_x, quantile_y, quantile_ang)



def scanline_lane(lanenum, linepts):

    lanepts=linepts[linepts[:,0]==lanenum,1:] 
    lanepts = lanepts.reshape((6,1))    
    lanepts = np.squeeze(lanepts).reshape((-1,2))    
    scan_line=[]    
    
    # multiple segments
    for i in range(len(lanepts)-1):    
        start_pt=lanepts[i]    
        end_pt=lanepts[i+1]    
        seg_scanline=bresenham_line(start_pt, end_pt)    
        if len(scan_line)==0:    
            scan_line=seg_scanline    
        else:                               
            scan_line=np.r_[scan_line,seg_scanline[1:,:]]
    return scan_line


def upate_scanline_pts(all_lane_line_pts,fx_median,fy_median):
    
    row_num = 0
#     print(fx_median,fy_median)
    # from csv string to integer
    all_lane_line_pts = all_lane_line_pts.astype(np.int)
    
    for each_row in all_lane_line_pts:
        row_num += 1
        if row_num %2 == 0:
            # y coordinates
            all_lane_line_pts[row_num-1,1:] += int(fy_median)
        else:
            # x coordinates
            all_lane_line_pts[row_num-1,1:] += int(fx_median)
    
    return all_lane_line_pts        



def draw_Scanline(shifted_scanline_pts,frame,color):
     
    lane=np.unique(shifted_scanline_pts[:,0])
    if color =='r':
        color_value = (0,0,255)
    if color =='b':
        color_value = (255,0,0)
    if color == 'g':
        color_value = (0,255,0)
    for lane_num in lane:
        line_pts = shifted_scanline_pts[shifted_scanline_pts[:,0] == lane_num,1:].astype(np.int)
        for col in range(line_pts.shape[1]-1): 
            point_x1, point_y1 = line_pts[:,col] 
            point_x2, point_y2 = line_pts[:,col+1] 
            lineThickness=2 
            new_frame = cv2.line(frame, (point_x1, point_y1), (point_x2, point_y2), color_value, lineThickness)
    return new_frame    
    
    
    
if __name__ == '__main__':


    video_fn = sys.argv[1]
    csvfile_path = sys.argv[2]

    cap = cv2.VideoCapture(video_fn)
    ret, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    h,w = prev_gray.shape
    
    select_roi = False
    
    cv2.imwrite(video_fn[:-4]+"frame0.jpg", prev)
    frame_num = 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("frameRate: ", fps)

    # Select ROI
    if select_roi == True:
        fromCenter = False
        r = cv2.selectROI("Img", prev, fromCenter,False)
    else:    
        r = [0,0,w,h]
    print(r)
    
    cv2.waitKey(0) # press enter
    if cv2.getWindowProperty("Img",1) < 0:
        cv2.destroyAllWindows()
    
    # pre_gray_crop
    pre_gray_crop = prev_gray[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    
    # scanline points
    # csvfile_path = os.path.join("scanline_folder",videoName+'_'+'scanlinePts.csv')
    dense_optical_flow = True
    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    # get frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    #define duration
    duration = 2.4
    max_frames = fps*duration*60
    
    all_lane_line_pts=np.genfromtxt(csvfile_path, dtype='unicode',delimiter=',')  
    all_lane_line_pts =  all_lane_line_pts[1:,:]
    all_lane_line_pts.astype(np.int)
    
    fx_accumulate, fy_accumulate = 0,0
    
    # Define the codec for output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')    
    
    # Set up output video
    if(2*w > 1920):
        out = cv2.VideoWriter('video_out_stabilized.avi', fourcc, fps, (round(w/2), round(h/2)))
    else:
        out = cv2.VideoWriter('video_out_stabilized.avi', fourcc, fps, (2*w, h))
        
#     with open(output_file_name, 'w') as f:
#         f.write("x_vector_25"+','+"y_vector_25"+","+"x_vector_50"+','+"y_vector_50"+","
#                 +"x_vector_75"+','+"y_vector_75"+","+"x_vector_90"+','+"y_vector_90"+","+"angle_25"+
#                 ","+"angle_50"+","+"angle_75"+","+"angle_90"+'\n')
#         f.flush()
#         f.close()
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    
    while frame_num < max_frames:
        
        ret, img = cap.read()
        
        curr_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        # Crop image
        curr_gray_crop = curr_gray[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    
        frame_num += 1
        
        # https://docs.opencv.org/3.3.0/dc/d6b/group__video__track.html
        flow = cv2.calcOpticalFlowFarneback(pre_gray_crop, curr_gray_crop, None, pyr_scale = 0.5, levels=3, winsize = 15
                                            ,iterations = 10, poly_n = 5, poly_sigma = 1.2, flags = 0)

        (fx_shifted,fy_shifted,quantile_x,quantile_y, quantile_ang) = flow_adjust(flow)    

#       print(fx_median,fy_median)
        
        pre_gray_crop = curr_gray_crop
        
        # calculate the accumated motion
        fx_accumulate += fx_shifted
        fy_accumulate += fy_shifted 
        
        
        shifted_scanline_pts = upate_scanline_pts(all_lane_line_pts,fx_accumulate,fy_accumulate) 
        
        if select_roi == False:
            flow_vis = draw_flow(img, flow, step=16)
            
#          flow_vis=cv2.resize(flow_vis,(round(w/2), round(h/2)))
#          cv2.imshow('flow',flow_vis)
        
        img = draw_Scanline(all_lane_line_pts,img,'b')
        
        img_after = draw_Scanline(shifted_scanline_pts,img,'r')
        
        cv2.putText(img_after,'blue: original scanline',(30,100), font, 1.5,(255,0,0),2,cv2.LINE_AA)
        cv2.putText(img_after,'red:  adjusted scanline',(30,150), font, 1.5,(0,0,255),2,cv2.LINE_AA)
        
        print("frame number: ", frame_num)
        
        # Write the frame to the file    
#         frame_out = cv2.hconcat([img_before, img_after])

        # If the image is too big, resize it.
#         if(frame_out.shape[1] > 1920): 
#             frame_out = cv2.resize(frame_out, (round(frame_out.shape[1]/4), round(frame_out.shape[0]/2))); 
        
        img_after=cv2.resize(img_after,(round(w/2), round(h/2)))
        
        cv2.imshow("before After",img_after)
        
#       out.write(img_after)    
        
        key = cv2.waitKey(10) # change the value from the original 0 (wait forever) to something appropriate    
        if key == 27:
            print('ESC')
            break    
        if cv2.getWindowProperty("before After",1) < 0:        
            break        
             
    cap.release()
    out.release()     
    cv2.destroyAllWindows()