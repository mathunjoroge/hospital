<style type="text/css">
.nav-side-menu {
overflow: auto;
font-family: verdana;
font-size: 12px;
font-weight: 200;
background-color: #2e353d;
position: fixed;
top: 0px;
width: 200px;
height: 100%;
color: #e1ffff;
}
.nav-side-menu .brand {
background-color: #23282e;
line-height: 50px;
display: block;
text-align: center;
font-size: 14px;
}
.nav-side-menu .toggle-btn {
display: none;
}
.nav-side-menu ul,
.nav-side-menu li {
list-style: none;
padding: 0px;
margin: 0px;
line-height: 35px;
cursor: pointer;
/*    
.collapsed{
.arrow:before{
font-family: FontAwesome;
content: "\f053";
display: inline-block;
padding-left:10px;
padding-right: 10px;
vertical-align: middle;
float:right;
}
}
*/
}
.nav-side-menu ul :not(collapsed) .arrow:before,
.nav-side-menu li :not(collapsed) .arrow:before {
font-family: FontAwesome;
content: "\f078";
display: inline-block;
padding-left: 10px;
padding-right: 10px;
vertical-align: middle;
float: right;
}
.nav-side-menu ul .active,
.nav-side-menu li .active {
border-left: 3px solid #d19b3d;
background-color: #4f5b69;
}
.nav-side-menu ul .sub-menu li.active,
.nav-side-menu li .sub-menu li.active {
color: #d19b3d;
}
.nav-side-menu ul .sub-menu li.active a,
.nav-side-menu li .sub-menu li.active a {
color: #d19b3d;
}
.nav-side-menu ul .sub-menu li,
.nav-side-menu li .sub-menu li {
background-color: #181c20;
border: none;
line-height: 28px;
border-bottom: 1px solid #23282e;
margin-left: 0px;
}
.nav-side-menu ul .sub-menu li:hover,
.nav-side-menu li .sub-menu li:hover {
background-color: #020203;
}
.nav-side-menu ul .sub-menu li:before,
.nav-side-menu li .sub-menu li:before {
font-family: FontAwesome;
content: "\f105";
display: inline-block;
padding-left: 10px;
padding-right: 10px;
vertical-align: middle;
}
.nav-side-menu li {
padding-left: 0px;
border-left: 3px solid #2e353d;
border-bottom: 1px solid #23282e;
}
.nav-side-menu li a {
text-decoration: none;
color: #e1ffff;
font-size: 1.5em;
font-style: bold;
}
.nav-side-menu li a i {
padding-left: 10px;
width: 20px;
padding-right: 20px;
}
.nav-side-menu li:hover {
border-left: 3px solid #d19b3d;
background-color: #4f5b69;
-webkit-transition: all 1s ease;
-moz-transition: all 1s ease;
-o-transition: all 1s ease;
-ms-transition: all 1s ease;
transition: all 1s ease;
}
@media (max-width: 767px) {
.nav-side-menu {
position: relative;
width: 100%;
margin-bottom: 10px;
}
.nav-side-menu .toggle-btn {
display: block;
cursor: pointer;
position: absolute;
right: 10px;
top: 10px;
z-index: 10 !important;
padding: 3px;
background-color: #ffffff;
color: #000;
width: 40px;
text-align: center;
}
.brand {
text-align: left !important;
font-size: 22px;
padding-left: 20px;
line-height: 50px !important;
}
}
@media (min-width: 767px) {
.nav-side-menu .menu-list .menu-content {
display: block;
}
#main {
width:calc(100% - 300px);
float: right;
}
}

</style>
<div class="nav-side-menu">
<div class="brand">&nbsp;</div>
 <i class="fa fa-bars fa-2x toggle-btn" id="close_bars" data-toggle="collapse" data-target="#menu-content"></i>
<div class="menu-list">
<ul id="menu-content" class="menu-content collapse out">
    <li>&nbsp;</li>
<li>&nbsp;&nbsp;<a href='index.php?search= &response=0'>Home</a></li>
<li>&nbsp;&nbsp;<a href='from_lab.php?search=&nbsp;&response=0'>patients from lab</a></li>
<li>&nbsp;&nbsp;<a href='pending.php'>pending lab results</a></li>
<li>&nbsp;&nbsp;<a href='history.php?search=0'>patient history</a></li>
<li>&nbsp;&nbsp;<a href='discharge.php?search=0&response=0'>discharge patient</a></li>
<li>&nbsp;&nbsp;<a href='inpatient.php?search= &response=0'>inpatients</a></li>
<li>&nbsp;&nbsp;<a href='theatre.php'>theatre list</a></li>
<li>&nbsp;&nbsp;<a href='summary.php'>surgery summaries</a></li>
<li>&nbsp;&nbsp;<a href='../icd/index.php'>diseases ref</a></li>
<li>&nbsp;&nbsp;<a href='emergency.php?search=%20&response=0'>emergency</a></li>
<li>&nbsp;&nbsp;<a href='oncology.php?code=<?php echo rand(); ?>'>oncology</a></li>
<li>&nbsp;&nbsp;<a  rel="facebox" href='drug_add.php'>add drug to list</a></li>
</ul>   
</div>
</div>
<div class="container" id="main">
<div class="row">