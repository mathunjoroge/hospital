<link rel="shortcut icon" href="favicon.ico">
<link rel="stylesheet" type="text/css" href="../fonts/font-awesome-4.7.0/css/font-awesome.min.css">
 <script>
$("thead[class='']").addClass("bg-primary");

 </script>
<script>
    $("close_bars").click(function() { 
	// Your code 
});
</script>
<style type="text/css">
#close_bars{
    
      
    }
       @media (max-width:629px) {
  img#logo {
    display: none;
   
  }
 #logo_mobile {
    display: block;
    max-width: 100%;
 }

}
  @media (min-width:629px) {
  img#logo_mobile {
    display: none;
   
  }
  img#logo {
    float:left;
    margin-left:-34.2%;
   
  }
  .jumbotron{
      margin-left:-10%;
      height:100%;
  }
  .body{
      background-color: #3786d6;
  }
  }

}
img#logo_mobile{
   
    width: 100%;
  
}
    body {
    text-transform: capitalize;
}


}
@media screen and (max-width: 600px) {
  #nav_lable {
    visibility: hidden;
    display: none;
  }
#top{ 
    position: fixed;
    width: 100%;
    

}
.jumbotron { 
    margin-left: -7%;
}
</style>
<style>
#result {
    height:20px;
    font-size:16px;
    font-family:Arial, Helvetica, sans-serif;
    color:#333;
    padding:5px;
    margin-bottom:10px;
    background-color:#FFFF99;
}
#country{
    border: 1px solid #999;
    background: #EEEEEE;
    padding: 5px 10px;
    box-shadow:0 1px 2px #ddd;
    -moz-box-shadow:0 1px 2px #ddd;
    -webkit-box-shadow:0 1px 2px #ddd;
}
.suggestionsBox {
    position: absolute;
    left: 21.2%;
    margin: 0;
    width: 268px;
    top: 32%;
    padding:0px;
    background-color: blue;
    color: #fff;
}
@media (max-width: 480px) {
  .suggestionsBox {
    position: absolute;
    left: 0%;
    margin: 0;
    width: 60%;
    top: 37%;
    padding:0px;
    background-color: blue;
    color: #fff;
}
}
.suggestionList {
    margin: 0px;
    padding: 0px;
}
.suggestionList ul li {
    list-style:none;
    margin: 0px;
    padding: 6px;
    border-bottom:1px dotted #666;
    cursor: pointer;
}
.suggestionList ul li:hover {
    background-color: #FC3;
    color:#000;
}


.load{
background-image:url(loader.gif);
background-position:right;
background-repeat:no-repeat;
}

#suggest {
    position:relative;
}
#view_as {
    position: absolute;
    left: 85.2%;
     margin-top: -5%;
}
.combopopup{
    padding:3px;
    width:268px;
    border:1px #CCC solid;    
}

</style> 
<style>
#result2 {
    height:20px;
    font-size:16px;
    font-family:Arial, Helvetica, sans-serif;
    color:#333;
    padding:5px;
    margin-bottom:10px;
    background-color:#FFFF99;
}
#country2{
    border: 1px solid #999;
    background: #EEEEEE;
    padding: 5px 10px;
    box-shadow:0 1px 2px #ddd;
    -moz-box-shadow:0 1px 2px #ddd;
    -webkit-box-shadow:0 1px 2px #ddd;
}
.suggestionsBox2 {
    position: absolute;
    left: 21.2%;
    margin: 0;
    width: 268px;
    top: 32%;
    padding:0px;
    background-color: blue;
    color: #fff;
}
@media (max-width: 480px) {
  .suggestionsBox2 {
    position: absolute;
    left: 0%;
    margin: 0;
    width: 60%;
    top: 37%;
    padding:0px;
    background-color: blue;
    color: #fff;
}
}
.suggestionList2 {
    margin: 0px;
    padding: 0px;
}
.suggestionList2 ul li {
    list-style:none;
    margin: 0px;
    padding: 6px;
    border-bottom:1px dotted #666;
    cursor: pointer;
}
.suggestionList2 ul li:hover {
    background-color: #FC3;
    color:#000;
}
ul {
    font-family:Arial, Helvetica, sans-serif;
    font-size:11px;
  
    padding:0;
    margin:0;
}

.load{
background-image:url(loader.gif);
background-position:right;
background-repeat:no-repeat;
}

#suggest2 {
    position:relative;
}

.combopopup{
    padding:3px;
    width:268px;
    border:1px #CCC solid;    
}

</style> 
<style type="text/css">
#logged{
    float:right;
    margin-right:-364px;
}
}
    @media (max-width:629px) {
  img#logo {
    display: none;
  }

}

@media screen and (max-width: 600px) {
  #nav_lable {
    visibility: hidden;
    display: none;
  }
  #view_as{
    margin-top: -2%;

  }
}
#logo {
    float:left;
}
</style>

    <div class="container" id="top" style="background-color: #3786d6;" ><img id="logo"  src="../logo.png" style="height:auto;" alt="M&M Caresoft"><img id="logo_mobile" src="../mobile-min.JPG"  alt="M&M Caresoft"><strong id="logged"  ><i class="fa fa-user">&nbsp;</i><?php echo $_SESSION['SESS_FIRST_NAME']; ?>&nbsp;<a href="../logout.php"><i style="color: red;" class="fa fa-power-off"></i><font  style="color: white;"> Log out</font></strong></a></li></div> 
    <p>&nbsp;</p>
    <p>&nbsp;</p>
    <?php
    $position=$_SESSION['SESS_LAST_NAME'];
    if (isset($_GET['page'])) {
       $_SESSION['view_as']=$_SESSION['SESS_LAST_NAME'];
    }
     ?>
    <?php
    
    if ($position=="doctor") {
        # code...
    
     ?>
    <style type="text/css">
            body {
  background-image: url('../images/doctor.jpg');
   background-repeat: no-repeat;
  background-attachment: fixed;
  background-position: center; 
 
}
        </style>
 <?php }?>
  <?php
    if ($position=="pharmacist") {
        # code...
    
     ?>
    <style type="text/css">
            body {
  background-image: url('../images/pharmacy.jpg');
   background-repeat: no-repeat;
  background-attachment: fixed;
  background-position: center; 
 
}
        </style>
 <?php }?>
   <?php
        if ($position=="lab") {
        # code...
    
     ?>
    <style type="text/css">
            body {
  background-image: url('../images/lab.jpg');
   background-repeat: repeat;
  background-attachment: fixed;
  background-position: center; 
 
}
        </style>
 <?php }?>
    <?php
       if ($position=="admin") {
        # code...
    
     ?>
    <div id="view_as" style="float: right">
    <p>
    <form action="../redirect.php" method="POST">
                    <label id="nav_lable"> <?php echo $_SESSION["view_as"]; ?>'s view, change to: </label>
                    <script type="text/javascript">
                        function getval(sel)
                    {
                   document.getElementById("submitbtn").click();
                   }
                    </script>
                    <select name="position" title="please select user" onchange="getval(this);" required/>
                    <option></option>
                        <option value="registration">records</option>
                        <option value="cashier">cashier</option>
                        <option value="nurse">nurse</option>
                        <option value="doctor">doctor</option>
                        <option value="pharmacist">pharmacist</option>
                        <option value="stores">store</option>
                        <option value="lab">lab</option>
                        <option value="imaging">imaging</option>
                        <option value="admin">admin</option>                        
                    </select>
                    <button id="submitbtn" style="display: none;">submit</button>
                </form></p></div>
            <?php } ?>

  