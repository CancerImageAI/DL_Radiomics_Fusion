# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:19:22 2015

@author: tizita nesibu
"""
import xml.etree.ElementTree as ET
import os, sys

from annotstructs import NoduleRoi, NormalNodule, SmallNodule, NonNodule, RadAnnotation

#RadAnnotation holds -> readingSession(is the annotaion of one raiologist slice by slice)
# unblindedReadNodule -> holds one slice's annotation info -> #if no characterstics -> SmallNodule
                                                              #if with characterstics -> NormalNodule
#NonNodule -> if it is not unblindedReadNodule 
#locus is like edgmap  for -> #NonNodule
                             
class LIDCXmlHeader:
    
    def __init__(self):       #4 elements are not included b/c they don't have data inside
        self.version = None 
        self.messageid = None
        self.date_request = None
        self.time_request = None
        self.task_descr = None
        self.series_instance_uid = None
        self.date_service = None
        self.time_service = None
        self.study_instance_uid = None
    
    def __str__(self):
        strng = ("--- XML HEADER ---\n"
                "Version (%s) Message-Id (%s) Date-request (%s) Time-request (%s) \n"
                "Series-UID (%s)\n"
                "Time-service (%s) Task-descr (%s) Date-service (%s) Time-service (%s)\n"
                "Study-UID (%s)")%(self.version, self.messageid, self.date_request, self.time_request,
                     self.series_instance_uid, self.time_service, self.task_descr, 
                     self.date_service, self.time_service, self.study_instance_uid)
        return strng
                

class LIDCXmlParser:
    
    def __init__(self, fname=[]):
        
        #check if file exists or not
        self.initialized = False
        if (not (fname == [])): #if fname is not empity
            if not os.path.isfile(fname):
                print("Error: filename (%s) doesn't exist"%(fname))
                self.initialized = False
            else:
                self.initialized = True

        self.xml_fname = fname
        self.rad_annotations = []   #to hold list of readingSession(xml element)->which holds each radiologists 
                            #annotation info i.e. len(rad_annotations) tells us nbr of radiologist 
        self.xml_header = LIDCXmlHeader()  
        self.namespace = {'nih': 'http://www.nih.gov'} #dict to store namespace's key and value b/c when this xml file 
                        #is parsed it containes this website infront of each tag that is parsed, to avoid including the
                        # whole site, namespace could be used to shorten it(can be indicated by the key i.e.'nih').
        
        if (self.initialized):
            print("LIDC Xml Parser Initialized!")
        return

    def set_xml_file(self, fname):
        #check if file exists or not
        if not os.path.isfile(fname):
            print("Error: filename (%s) doesn't exist"%(fname))
            self.initialized = False
        else:
            self.xml_fname = fname
            self.initialized = True
        
        return self.initialized
        
    def parse(self):
        if (not self.initialized): # if file not exist(if self.initialized is false)
            print("Error: Parser not initiialized!")
            return
        ns = self.namespace
        
        tree = ET.parse(self.xml_fname) #ET is the library we use to parse xml data
        root = tree.getroot()
        
        #print root[0][0].tag, root[0][0].text
        #print root[0][1].tag, root[0][1].text
        #print root[0][2].tag, root[0][2].text
        
        #print root.attrib
        #parse the file
        
        #FIXME: Exception Handling        
        resp_hdr = root.findall('nih:ResponseHeader', ns)[0]  #ns is to show what nih is,and [0] is b/c only one ResponseHeader is available
   #4 elements are not included b/c they don't have data inside
        if resp_hdr.find('nih:Version', ns) is not None:
            self.xml_header.version = resp_hdr.find('nih:Version', ns).text
        if resp_hdr.find('nih:MessageId', ns) is not None:
            self.xml_header.messageid = resp_hdr.find('nih:MessageId', ns).text
        if resp_hdr.find('nih:DateRequest', ns) is not None:
            self.xml_header.date_request = resp_hdr.find('nih:DateRequest', ns).text
        if resp_hdr.find('nih:TimeRequest', ns) is not None:
            self.xml_header.time_request = resp_hdr.find('nih:TimeRequest', ns).text
        if resp_hdr.find('nih:TaskDescription', ns) is not None:
            self.xml_header.task_descr = resp_hdr.find('nih:TaskDescription', ns).text
        if resp_hdr.find('nih:SeriesInstanceUid', ns) is not None:
            self.xml_header.series_instance_uid = resp_hdr.find('nih:SeriesInstanceUid', ns).text
        if resp_hdr.find('nih:DateService', ns) is not None:
            self.xml_header.date_service = resp_hdr.find('nih:DateService', ns).text
        if resp_hdr.find('nih:TimeService', ns) is not None:
            self.xml_header.time_service = resp_hdr.find('nih:TimeService', ns).text
        if resp_hdr.find('nih:StudyInstanceUID', ns) is not None:
            self.xml_header.study_instance_uid = resp_hdr.find('nih:StudyInstanceUID', ns).text
        
        print(self.xml_header) # calles str of the class LIDCXmlHeader() 
            

            
        for read_session in root.findall('nih:readingSession',ns): #readingSession-> holds radiologist's annotation info
            rad_annotation = RadAnnotation() #to hold each radiologists annotation i.e. readingSession in xml file
            rad_annotation.version = read_session.find('nih:annotationVersion', ns).text            
            rad_annotation.id = read_session.find('nih:servicingRadiologistID', ns).text
            
            unblinded_nodule = read_session.findall('nih:unblindedReadNodule', ns)
            
            for node in unblinded_nodule:
                nodule = self.parse_nodule(node)
                
#                if (not nodule.is_small):
#                    rad_annotation.nodules.append(nodule)
#                else:
#                    rad_annotation.small_nodules.append(nodule)
#                    
                if(nodule.is_small):
                    rad_annotation.small_nodules.append(nodule)
                else:
                    rad_annotation.nodules.append(nodule) # nodule is normalNodule
                    

            non_nodule = read_session.findall('nih:nonNodule', ns)
            
            for node in non_nodule:
                nodule = self.parse_non_nodule(node)
                rad_annotation.non_nodules.append(nodule)
           
            self.rad_annotations.append(rad_annotation)
        
        return
        #for child in root:
        #    print child.tag#, child.attrib
            
    def parse_nodule(self, xml_node): #xml_node is one unblindedReadNodule
        ns = self.namespace
        
        chartcs_node = xml_node.find('nih:characteristics', ns)
        is_small = (chartcs_node is None) # if no characteristics, it is smallnodule  i.e. is_small=TRUE
        
        if (is_small) or (chartcs_node.find('nih:subtlety',ns) is None):
            nodule = SmallNodule()
            nodule.id = xml_node.find('nih:noduleID', ns).text
        else:
            nodule = NormalNodule() #if it has characteristics it is normalNodule
            nodule.id = xml_node.find('nih:noduleID', ns).text

            nodule.characterstics.subtlety = int(chartcs_node.find('nih:subtlety',ns).text)
            nodule.characterstics.internalstructure = int(chartcs_node.find('nih:internalStructure',ns).text)
            nodule.characterstics.calcification = int(chartcs_node.find('nih:calcification',ns).text)
            nodule.characterstics.sphericity = int(chartcs_node.find('nih:sphericity',ns).text)
            nodule.characterstics.margin = int(chartcs_node.find('nih:margin',ns).text)
            nodule.characterstics.lobulation = int(chartcs_node.find('nih:lobulation',ns).text)
            nodule.characterstics.spiculation = int(chartcs_node.find('nih:spiculation',ns).text)
            nodule.characterstics.texture = int(chartcs_node.find('nih:texture',ns).text)
            nodule.characterstics.malignancy = int(chartcs_node.find('nih:malignancy',ns).text)
            
        xml_rois = xml_node.findall('nih:roi', ns)

        for xml_roi in xml_rois:
            roi = NoduleRoi()
            roi.z = float(xml_roi.find('nih:imageZposition', ns).text)
            roi.sop_uid = xml_roi.find('nih:imageSOP_UID', ns).text
            roi.inclusion = (xml_roi.find('nih:inclusion', ns).text == "TRUE") # when inclusion = TRUE ->roi includes the whole nodule
                                            #when inclusion = FALSE ->roi is drown twice for one nodule 1.ouside the nodule
                                            #2.inside the nodule -> to indicate that the nodule has donut hole(the inside hole is 
                                            #not part of the nodule) but by forcing inclusion to be TRUE, this situation is ignored
                
            edgemaps =  xml_roi.findall('nih:edgeMap', ns)

            xmin, xmax, ymin, ymax = sys.maxsize,0,sys.maxsize,0  #???????????????????
            
            for edgemap in edgemaps:
                x = int(edgemap.find('nih:xCoord', ns).text)
                y = int(edgemap.find('nih:yCoord', ns).text)
                
                if (x > xmax):   # to define a rectangle arround the roi 
                    xmax = x     #only the 1st point i.e.(xmin, ymin) and 
                                 #the last point(xmax, ymax) is needed to drow a rectangle
                if (x < xmin):
                    xmin = x
                    
                if (y > ymax):
                    ymax = y
               
                if (y < ymin):
                   ymin = y
                
                
                roi.roi_xy.append((x,y))
     
            if not is_small:   #only for normalNodules
                roi.roi_rect = (xmin, ymin, xmax, ymax)
                roi.roi_centroid = ((xmax+xmin)/2., (ymin+ymax)/2.) #center point 

            nodule.rois.append(roi)
                
        return nodule  #is equivalent to unblindedReadNodule(xml element)

    def parse_non_nodule(self, xml_node):   #xml_node is one nonNodule
        ns = self.namespace
        
        nodule = NonNodule()        

        nodule.id = xml_node.find('nih:nonNoduleID', ns).text
        roi = NoduleRoi()
        roi.z =  float(xml_node.find('nih:imageZposition', ns).text)
        roi.sop_uid =  xml_node.find('nih:imageSOP_UID', ns).text

        loci =  xml_node.findall('nih:locus', ns)
        
        for locus in loci:
            x = int(locus.find('nih:xCoord', ns).text)
            y = int(locus.find('nih:yCoord', ns).text)
            roi.roi_xy.append((x,y))
        nodule.rois.append(roi)    
        return nodule    #is equivalent to nonNodule(xml element)
      
    
    def __str__(self):   #to print the whole xml data of a patient(not important)
        strng = "*"*79 + "\n"
        strng += "XML FileName [%s] \n"%self.xml_fname
        strng += str(self.xml_header) #str calles LIDCXmlHeader's str b/c xml_header is object of LIDCXmlHeader class
        
        strng += "# of Rad. Annotations [%d] \n" % len(self.rad_annotations)
        
        for ant in self.rad_annotations:
            strng += str(ant)
        
        strng += "*"*79 + "\n"  
        return strng
         
        

#def main():
#    dt = LIDCXmlParser(r'F:\ImageData\LIDC\DOI\LIDC-IDRI-0001\1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178\1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192\069.xml')
#    dt.parse()
#    print(dt)
#    return


#if __name__ == '__main__':
##    if __package__ is None:
##        path_abs_name = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
##        if (not (path_abs_name in set(os.sys.path))):
##            os.sys.path.append(path_abs_name)
##
##        from structs.annotstructs import NoduleRoi, NormalNodule, SmallNodule, NonNodule, RadAnnotation
##    else:
##        from ..structs.annotstructs import NoduleRoi, NormalNodule, SmallNodule, NonNodule, RadAnnotation
#    dt = LIDCXmlParser(r'F:\ImageData\LIDC\DOI\LIDC-IDRI-0001\1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178\1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192\069.xml')
#    dt.parse()
#    print(dt)



